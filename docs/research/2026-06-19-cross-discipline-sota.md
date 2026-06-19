# Cross-discipline SOTA for CF-LIBS — verified knowledge base

**Date:** 2026-06-19 · **Source:** Phase-2 cross-discipline literature sweep (8 agents) → asta-grounded verification (Semantic Scholar): **62 citations confirmed, 26 DOI-corrected, 0 unverifiable**, + **35 new 2019-2026 methods** found via `asta papers`/citation-graph walking.

**123 methods total; 110 are physics-only-compatible** (no sklearn/torch/tf/jax.nn — eligible for the shipped pipeline). Methods marked `[NEW]` were surfaced by the asta augmentation pass; `[phys✗]` violate the physics-only constraint and are reference-only.

> Citation status from asta: `confirmed` = title/year/venue/DOI matched; `corrected` = DOI/detail fixed against Semantic Scholar; all entries below carry a resolvable DOI or S2 id.

## Calibration-free LIBS quantification: survey of CF-LIBS variants and their reported accuracy/precision, and which methods are most likely to beat a standard Saha-Boltzmann CF-LIBS inversion

### Classic CF-LIBS (Ciucci-Corsi-Palleschi algorithm) - the baseline
- **Applicability:** inversion-quantification (the standard pipeline: Boltzmann-plot T, Saha n_e, closure-fixed scale factor) and forward model
- **Expected benefit:** Reference baseline; unguarded classic CF typically ~10-20% relative error on majors. Everything else is measured against this.
- **Integration difficulty:** low
- **Citation:** *New Procedure for Quantitative Elemental Analysis by Laser-Induced Plasma Spectroscopy*, Applied Spectroscopy (S2 venue field empty; DOI confirms Appl. Spectrosc. 53(8):960-964), 1999 — doi:10.1366/0003702991947612

### Tognoni et al. numerical accuracy/precision study (intrinsic error floor)
- **Applicability:** uncertainty / error-budget benchmark for the whole inversion-quantification stage
- **Expected benefit:** Quantifies the intrinsic floor: even ideal CF-LIBS error is dominated by T precision; argues for wide E_k lever-arm, multi-element common-slope, per-element uncertainty reporting.
- **Integration difficulty:** low
- **Citation:** A numerical study of expected accuracy and precision in Calibration-Free Laser-Induced Breakdown Spectroscopy in the assumption of ideal analytical plasma, Spectrochimica Acta Part B (62(12):1287-1302), 2007 — doi:10.1016/J.SAB.2007.10.005

### Multi-element Saha-Boltzmann plot (Aguilera & Aragon)
- **Applicability:** inversion-quantification: temperature-determination core of a robust CF inversion (pooled multi-species fit)
- **Expected benefit:** Reduces the dominant T error by widening E_k span and pooling lines across species; strict improvement over per-species Boltzmann plots.
- **Integration difficulty:** low
- **Citation:** *Multi-element Saha-Boltzmann and Boltzmann plots in laser-induced plasmas*, Spectrochimica Acta Part B (62(4):378-385), 2007 — doi:10.1016/J.SAB.2007.03.024

### One-point calibration CF-LIBS (OPC) - best reported trueness among standardless-adjacent methods `[DOI-corrected]`
- **Applicability:** inversion-quantification + spectral-response/uncertainty: a bias-correction layer wrapping the standard Saha-Boltzmann inversion
- **Expected benefit:** Best documented trueness of all reviewed CF variants (<1 wt% on majors, bronzes); ~15% avg uncertainty vs ~53% for inverse-CF without SA correction. CORRECTION: survey DOI 10.1016/j.sab.2013.05.002 is wrong; correct DOI is 10.1016/J.SAB.2013.05.016 (authors Cavalcanti, Teixeira, Legnaioli, Lorenzetti, Pardini, Palleschi confirmed).
- **Integration difficulty:** medium
- **Citation:** *One-point calibration for calibration-free laser-induced breakdown spectroscopy quantitative analysis*, Spectrochimica Acta Part B (87:51-56), 2013 — doi:10.1016/J.SAB.2013.05.016

### One-point calibration of the Saha-Boltzmann plot (extends OPC, single-line capable) `[DOI-corrected]`
- **Applicability:** inversion-quantification: hardening the Saha-Boltzmann intercept/scale step against shot-to-shot and matrix variation
- **Expected benefit:** Reported accuracy AND precision gains over both classic CF and plain OPC; single-line-per-element capable. CORRECTION: survey attributed it to 'Praher / Cavalcanti-lineage, 2018, DOI 10.1016/j.sab.2018.09.022' - all wrong. Actual paper is Borduchi, Milori, Villas-Boas, SAB 2019, DOI 10.1016/j.sab.2019.105692.
- **Integration difficulty:** medium
- **Citation:** One-point calibration of Saha-Boltzmann plot to improve accuracy and precision of quantitative analysis using laser-induced breakdown spectroscopy, Spectrochimica Acta Part B: Atomic Spectroscopy, 2019 — doi:10.1016/j.sab.2019.105692

### C-sigma (CSigma) graphs - generalized curve-of-growth, self-absorption-tolerant (Aragon & Aguilera) `[DOI-corrected]`
- **Applicability:** self-absorption + inversion-quantification + forward model: a self-absorption-native alternative to the Boltzmann-plot inversion that yields T and composition together
- **Expected benefit:** C-sigma avg relative error <10% on fused-glass rock; principled radiative-transfer SA treatment. Primary 2014 JQSRT paper CONFIRMED (DOI 10.1016/J.JQSRT.2014.07.026). CORRECTION: companion 'generalized curves of growth' paper (Aguilera, Aragon, Manrique, SAB 110, 2015) - survey DOI 10.1016/j.sab.2015.06.003 is wrong; correct DOI 10.1016/J.SAB.2015.06.010 (paperId 423231fa46cd7768d573246c29db68c8f23baf09).
- **Integration difficulty:** high
- **Citation:** *CSigma graphs: A new approach for plasma characterization in laser-induced breakdown spectroscopy*, Journal of Quantitative Spectroscopy and Radiative Transfer (149:90-102; corrigendum DOI 10.1016/J.JQSRT.2015.03.001 confirms pages), 2014 — doi:10.1016/J.JQSRT.2014.07.026

### Columnar-density Saha-Boltzmann (CD-SB) - exploits self-absorption instead of avoiding it (Cristoforetti & Tognoni) `[DOI-corrected]`
- **Applicability:** self-absorption + inversion-quantification: preferred T/composition route when resonance/strong lines dominate (late gates, high concentrations)
- **Expected benefit:** CD-SB relative error <4% on CaO in limestone; turns the worst (self-absorbed) lines into the most informative; removes detector-response dependence. CORRECTION: survey DOI 10.1016/j.sab.2012.11.002 is wrong; correct DOI 10.1016/J.SAB.2012.11.010.
- **Integration difficulty:** medium
- **Citation:** Calculation of elemental columnar density from self-absorbed lines in laser-induced breakdown spectroscopy: A resource for quantitative analysis, Spectrochimica Acta Part B (79-80:63-71), 2013 — doi:10.1016/J.SAB.2012.11.010

### Columnar-density + standard reference line (single-sample-calibrated CD-SB) - strong recent SA-corrected accuracy `[DOI-corrected]`
- **Applicability:** self-absorption + inversion-quantification: drop-in upgrade to the SA-correction + intercept-anchoring stages
- **Expected benefit:** Reported error reductions on Al-bronze ~3.2% -> ~1% and Al alloy to ~0.3-0.4%; among the best end-to-end CF-LIBS accuracy numbers. CORRECTION: survey cited Anal. Chim. Acta 1183:338991, 2021, DOI 10.1016/j.aca.2021.338991 - that DOI resolves to an unrelated aptasensor paper. The exact-titled paper is Deng, Hu, Zhang, Chen, Niu, Nie, Zeng, Guo, Optics Express 2022, DOI 10.1364/oe.446334 (Zhenlin Hu is an author, consistent with survey's 'Hu').
- **Integration difficulty:** medium
- **Citation:** *Accuracy improvement of single-sample calibration laser-induced breakdown spectroscopy with self-absorption correction*, Optics Express (30(8)), 2022 — doi:10.1364/oe.446334

### Internal-reference self-absorption correction (IRSAC and descendants)
- **Applicability:** self-absorption + line-selection: lightweight SA correction inside the standard inversion (no Stark coefficients needed)
- **Expected benefit:** Restores Boltzmann linearity; improves T and composition vs basic CF on Al and Fe-Cr/Fe-Cr-Ni alloys. Both Sun & Yu 2009 and Yang 2018 auto-select variant CONFIRMED.
- **Integration difficulty:** low
- **Citation:** *Correction of self-absorption effect in calibration-free laser-induced breakdown spectroscopy by an internal reference method*, Talanta (79(2):388-395); auto-select variant: Yang et al. 2018, DOI 10.1177/0003702817734293, Applied Spectroscopy, 2009 — doi:10.1016/j.talanta.2009.03.066

### Self-calibrated LIBS (SC-LIBS) under partial-LTE with blackbody normalization (De Giacomo group)
- **Applicability:** inversion-quantification + non-LTE handling: CF variant for broadband/time-integrated spectra where full LTE and clean closure are not safe (relevant to ChemCam/SuperCam data)
- **Expected benefit:** SC-LIBS trueness 'better than 5 wt%' on majors (meteorites); value is robustness to non-LTE and time-integration bias plus removal of closure-induced trace-element error.
- **Integration difficulty:** high
- **Citation:** *Self-Calibrated Laser-Induced Breakdown Spectroscopy for the Quantitative Elemental Analysis of Suspended Volcanic Ash*, Applied Spectroscopy (78(8)), 2024 — doi:10.1177/00037028241241076

### Inverse CF-LIBS (Gaudiuso) and 3D-CF-LIBS / time-resolved Boltzmann (Bredice) `[DOI-corrected]`
- **Applicability:** inversion-quantification + uncertainty: alternative T-determination routes (inverse uses a standard; 3D removes A_ki dependence)
- **Expected benefit:** Inverse CF trueness <6 wt% on Zn (bronzes) but ~53% avg uncertainty without SA correction; 3D's A_ki-independence is its real advantage. CORRECTIONS: (a) the 2012 inverse-CF DOI 10.1016/J.SAB.2012.06.034 is a real Gaudiuso/Dell'Aglio/De Pascale/De Giacomo paper but its TITLE is the copper-alloy LTE-approach one above, NOT 'LIBS of archaeological findings with calibration-free inverse method' (that title belongs to the 2014 Anal. Chim. Acta follow-up, DOI 10.1016/j.aca.2014.01.020). (b) the Bredice '3D-CF-LIBS' DOI 10.1016/j.sab.2020.105898 resolves to an unrelated Polish-banknotes XRF/LIBS paper; the real Bredice 2020 paper is 'Study of binary lead-tin alloys using a new procedure based on calibration-free LIBS', Urbina/Carneiro/Rocha/Farias/Bredice/Palleschi, SAB 2020, DOI 10.1016/j.sab.2020.105902.
- **Integration difficulty:** high
- **Citation:** Laser-induced plasma analysis of copper alloys based on Local Thermodynamic Equilibrium: An alternative approach to plasma temperature determination and archeometric applications, Spectrochimica Acta Part B (74-75:38-45), 2012 — doi:10.1016/J.SAB.2012.06.034

### Monte-Carlo / full-spectrum forward-fitting CF (Gornushkin & Voelker) - intrinsic-accuracy ceiling `[DOI-corrected]`
- **Applicability:** inversion-quantification (primary solver) + self-absorption + forward model: the line-based Boltzmann pipeline becomes the initializer/seed
- **Expected benefit:** Synthetic multi-element samples: intrinsic relative accuracy ~1%, ~10x better than Boltzmann-plot CF when model matches reality; GPU minutes/spectrum - natural fit for the repo JAX manifold path. CORRECTION: survey title 'Capabilities of Calibration-Free LIBS: A Monte Carlo study' is wrong; the DOI 10.3390/s22197149 (Gornushkin & Voelker, Sensors 2022) is real but titled 'Intrinsic Performance of Monte Carlo Calibration-Free Algorithm for LIBS'. Closed-form companion CONFIRMED: Voelker & Gornushkin, JAAS 2023, DOI 10.1039/d2ja00352j, 'Investigation of a Method for the Correction of Self-Absorption by Planck Function in LIBS' (the I_thin = -B ln(1-I/B) correction).
- **Integration difficulty:** high
- **Citation:** *Intrinsic Performance of Monte Carlo Calibration-Free Algorithm for Laser-Induced Breakdown Spectroscopy*, Sensors (22(19):7149), 2022 — doi:10.3390/s22197149

### Machine-learning-supported real-time CF-LIBS (knowledge only; NOT shippable here) `[phys✗]`
- **Applicability:** inversion-quantification + real-time: speed and non-LTE robustness, but as a learned surrogate not a physics solver
- **Expected benefit:** Near-instant T/n_e/composition; accuracy comparable to classical methods with good training coverage. Knowledge only - physics_only_compatible=false per the shipped-code constraint (sklearn/torch/jax.nn banned). CONFIRMED: DOI 10.1016/j.sab.2024.107082, SAB 2024 (Favre, Abad, Poux, Gosse, Berjaoui, Morel, Bultel). The survey's 'Borges et al. ANN-CF-LIBS' alternate citation was not separately resolved but the main 2024 ML paper is confirmed.
- **Integration difficulty:** high
- **Citation:** *Towards real-time calibration-free LIBS supported by machine learning*, Spectrochimica Acta Part B: Atomic Spectroscopy (107082), 2024 — doi:10.1016/j.sab.2024.107082

### Precomputed self-absorption correction + one-point calibration CF-LIBS (Qiu, Gornushkin et al. 2026) `[NEW]`
- **Applicability:** inversion-quantification + self-absorption: combines the survey's top-2 recommended upgrades (radiative-transfer SA correction + OPC bias removal) and PRECOMPUTES the SA correction, so it maps directly onto this repo's JAX/HDF5 manifold pre-computation path. Drop-in over the Saha-Boltzmann inversion + SA stages. Forward-citation of Gornushkin & Voelker 2022.
- **Expected benefit:** Newest (2026) demonstration that SA-correction + one-point anchoring is the highest-leverage physics-only combination, with the SA correction made cheap via precomputation (amortizes the C-sigma/CDSB cost the survey flagged as 'computationally heavier'). Directly actionable: the repo already has a manifold pre-compute stage to host the precomputed SA table.
- **Integration difficulty:** medium
- **Citation:** *Calibration-free LIBS improved by precomputed self-absorption correction and one-point calibration*, Spectrochimica Acta Part B: Atomic Spectroscopy (107520), 2026 — doi:10.1016/j.sab.2026.107520

### Genetic-algorithm-optimized plasma temperature CF-LIBS (Dong, Lu et al.) `[NEW]`
- **Applicability:** inversion-quantification (T-determination step): replace/augment the Boltzmann-plot slope with a global optimizer that searches T to maximize multi-species Saha-Boltzmann linearity / minimize spectral residual. The GA is a black-box stochastic optimizer (NOT a learned ML model), so it satisfies the physics-only constraint. Missed by the survey despite being a direct T-step upgrade and a cheaper alternative to full Monte-Carlo forward-fitting.
- **Expected benefit:** Reported accuracy improvement over slope-fit CF on alloys by escaping the local linear-regression bias of the Boltzmann plot; a lighter-weight bridge between line-based CF and the Gornushkin-Voelker full-spectrum stochastic optimizer. Physics-only and integrable as an alternative T-solver strategy under the repo's SolverStrategy ABC.
- **Integration difficulty:** medium
- **Citation:** A method for improving the accuracy of calibration-free laser-induced breakdown spectroscopy (CF-LIBS) using determined plasma temperature by genetic algorithm (GA), Journal of Analytical Atomic Spectrometry, 2015 — doi:10.1039/C4JA00470A

### Self-absorption correction by blackbody radiation reference (Hu, Zhang et al.) `[NEW]`
- **Applicability:** self-absorption + inversion-quantification: uses the plasma's own blackbody-like continuum as the optically-thick saturation reference to recover each line's true (optically-thin) intensity. Conceptually the empirical precursor to the Voelker-Gornushkin Planck-function closed-form correction (I_thin = -B ln(1-I/B)) the survey already lists. Missed by the survey but more cited (60) than several included methods and directly relevant to the SA-correction stage.
- **Expected benefit:** Restores Boltzmann linearity without per-line Stark coefficients; pairs with the repo's existing CDSB self-absorption module as an alternative continuum-anchored SA estimator. Physics-only (radiative-transfer based).
- **Integration difficulty:** medium
- **Citation:** *Correction of self-absorption effect in calibration-free laser-induced breakdown spectroscopy (CF-LIBS) with blackbody radiation reference*, Analytica Chimica Acta, 2019 — doi:10.1016/j.aca.2019.01.016

## Plasma modeling and LTE diagnostics state of the art for CF-LIBS forward modeling: Saha-Boltzmann vs collisional-radiative/non-LTE, LTE validity criteria beyond McWhirter, Stark broadening theory and databases, electron-density diagnostics, partition functions, and ionization-potential depression / Debye-Huckel corrections

### Temporal/relaxation LTE validity criterion beyond McWhirter (Cristoforetti et al. 2010)
- **Applicability:** forward model + uncertainty: gates whether Saha-Boltzmann LTE is physically applicable; should be evaluated per spectrum/gate-window as a validity flag feeding the uncertainty budget; canonical reference to make cflibs/plasma/lte_validator.py temporal check rigorous (e-collision thermalization rate vs dT/dt, dn_e/dt).
- **Expected benefit:** Prevents systematic bias from applying Saha-Boltzmann outside its validity domain; McWhirter threshold alone can pass while LTE is violated. Screening criterion (no single % figure); companion CR study quantifies avoided error at 10-42%. 609 citations.
- **Integration difficulty:** low
- **Citation:** *Local Thermodynamic Equilibrium in Laser-Induced Breakdown Spectroscopy: Beyond the McWhirter criterion*, Spectrochimica Acta Part B (65, 86-95; venue blank in S2 metadata but DOI confirms SAB 65 2010), 2010 — doi:10.1016/J.SAB.2009.11.005

### Zero-dimensional collisional-radiative (CR) model coupled to the electron Boltzmann equation (Pietanza et al. 2010)
- **Applicability:** forward model: a non-LTE alternative/benchmark for Saha-Boltzmann level populations; use offline to map the LTE-validity envelope (T, n_e, delay) and derive per-level departure coefficients b_k applied as a multiplicative correction to LTE populations in cflibs/plasma/saha_boltzmann.py without shipping a full CR solver.
- **Expected benefit:** Companion LTE-vs-CR comparisons report population deviations ~10% at n_e 1e18-1e19 cm^-3, ~27% at 1e17, ~42% at 1e15 cm^-3; quantifies the floor below which Saha-Boltzmann inversion is unsafe. 65 citations.
- **Integration difficulty:** high
- **Citation:** *Kinetic processes for laser induced plasma diagnostic: A collisional-radiative model approach*, Spectrochimica Acta Part B (65, 616-626; venue blank in S2, DOI confirms SAB 65 2010), 2010 — doi:10.1016/J.SAB.2010.03.012

### Comprehensive thermodynamic-equilibrium-state framework (pLTE, two-temperature, ionizing/recombining) (Cristoforetti, Tognoni, Gizzi 2013)
- **Applicability:** forward model + line selection: physics to weight/exclude lines whose lower level lies below the pLTE thermalization limit, and to flag ionizing/recombining conditions where the standard Saha ionic->neutral mapping is biased; feeds lte_validator and line_selection.
- **Expected benefit:** Reduces systematic Boltzmann-plot bias by restricting fits to thermalized levels; the pLTE-limit concept is the principled basis for E_lower line-selection cuts the inversion already does heuristically. Method-enabling. 135 citations.
- **Integration difficulty:** medium
- **Citation:** *Thermodynamic equilibrium states in laser-induced plasmas: From the general case to laser-induced breakdown spectroscopy plasmas*, Spectrochimica Acta Part B (90, 1-22; venue blank in S2, DOI confirms SAB 90 2013). NOTE: an erratum exists (SAB 2014, DOI 10.1016/J.SAB.2014.06.002, paperId 6579de20c41927eab65a91670aacebc3d83cb257)., 2013 — doi:10.1016/J.SAB.2013.09.004

### Uniform-LTE 'ideal radiation source' forward model with full spectral-radiance fitting (Hermann et al. 2017)
- **Applicability:** forward model + inversion (full-spectrum fitting): replaces/augments the discrete Boltzmann-plot pathway with a uniform-slab radiative-transfer spectrum fit over T, n_e, composition; maps onto the existing JAX SpectrumModel as the kernel of a least-squares/Bayesian inversion, folding self-absorption, instrument response and partition functions into one consistent forward computation.
- **Expected benefit:** Hermann's group reports CF-LIBS accuracies of order a few percent to sub-percent for majors on alloys/thin films when the uniform-LTE window is respected; removes the resonance-line-exclusion problem. Best-in-class forward-model fidelity for time-gated single-shot spectra. 39 citations.
- **Integration difficulty:** high
- **Citation:** *Ideal radiation source for plasma spectroscopy generated by laser ablation*, Physical Review E (96, 053210), 2017 — doi:10.1103/PhysRevE.96.053210

### Blackbody-limit diagnostic for LTE and absolute intensity calibration (Hermann et al. 2018)
- **Applicability:** forward model + uncertainty + intensity calibration: in-spectrum LTE proof and absolute-radiance anchor; implementable as a check that optically-thick line cores in the forward model do not exceed B_lambda(T), and as a route to retrieve the spectral-response E(lambda) currently absent from the inversion path.
- **Expected benefit:** Directly attacks the spectral-response systematic (response rotates the Boltzmann slope) by self-calibrating E(lambda); also a cheap, robust LTE validity check using only the strongest lines. Instrument-specific magnitude but removes an otherwise unconstrained systematic.
- **Integration difficulty:** medium
- **Citation:** *Local thermodynamic equilibrium in a laser-induced plasma evidenced by blackbody radiation*, Spectrochimica Acta Part B (144, 82-86), 2018 — doi:10.1016/J.SAB.2018.03.013

### Computer-simulated (ion-dynamics) Stark line-shape tables for hydrogen Balmer lines (Gigosos, Gonzalez, Cardenoso 2003)
- **Applicability:** electron-density diagnostic + forward model: reference width-n_e relations for the H-Balmer n_e channel; provides the non-linear width-n_e tables to replace the linear convention and the H-alpha ~n_e^0.7 exponent flagged as a known limitation in stark_ne.py.
- **Expected benefit:** Ion-dynamics-corrected H-beta gives n_e to ~5-10% vs larger errors from analytic Griem on H-alpha; correcting the n_e^0.7 vs n_e^1 scaling removes a systematic that propagates through the Saha equation into composition. 474 citations.
- **Integration difficulty:** low
- **Citation:** *Computer simulated Balmer-alpha, -beta and -gamma Stark line profiles for non-equilibrium plasmas diagnostics*, Spectrochimica Acta Part B (58, 1489-1504; venue blank in S2, DOI confirms SAB 58 2003), 2003 — doi:10.1016/S0584-8547(03)00097-1

### STARK-B (VAMDC) semiclassical-perturbation Stark width/shift database for non-hydrogenic lines (Sahal-Brechot et al. 2015)
- **Applicability:** forward model (line broadening) + electron-density diagnostic: populate the Stark-coefficient table consumed by cflibs/radiation broadening and stark_ne.py for non-H diagnostic lines (Ca II, Mg II, Fe, Al); provides T-dependence of widths the current single-coefficient convention approximates.
- **Expected benefit:** Replaces ad hoc / single-value Stark coefficients with theory-grade T-scaled widths for hundreds of lines; improves forward line shapes and Stark-width n_e robustness when H lines are weak/absent (common in metal/rock matrices). 53 citations.
- **Integration difficulty:** medium
- **Citation:** *The STARK-B database VAMDC node: a repository for spectral line broadening and shifts due to collisions with charged particles*, Physica Scripta (90, 054008; venue blank in S2, DOI confirms Phys. Scr. 90 2015), 2015 — doi:10.1088/0031-8949/90/5/054008

### NIST critically-evaluated experimental Stark widths and shifts (Konjevic, Lesage, Fuhr, Wiese 2002) `[DOI-corrected]`
- **Applicability:** forward model (broadening) + n_e diagnostic validation: graded experimental reference to validate/weight Stark coefficients in broadening and stark_ne.py; assigns per-line Stark-coefficient uncertainties propagating into n_e uncertainty.
- **Expected benefit:** Enables accuracy-weighted Stark coefficients and realistic n_e error bars; flags lines where theory-vs-experiment disagreement (tens of %) would silently bias n_e and the Saha correction. 542 citations. SUPERSEDED/EXTENDED by Djurovic-Blagojevic-Konjevic 2023 (see new_methods).
- **Integration difficulty:** low
- **Citation:** Experimental Stark Widths and Shifts for Spectral Lines of Neutral and Ionized Atoms (A Critical Review of Selected Data for the Period 1989 through 2000), Journal of Physical and Chemical Reference Data (31, 819-927). CORRECTION: survey DOI 10.1063/1.1486456 is invalid (404 in asta/Crossref); correct DOI is 10.1063/1.1525443. (S2 lists year 1990 -- a metadata quirk for old JPCRD reviews; actual publication is JPCRD vol 31, 2002. Title and author list confirmed.), 2002 — doi:10.1063/1.1525443

### Self-consistent, cutoff-aware partition functions (Barklem & Collet 2016)
- **Applicability:** forward model (Saha-Boltzmann normalization): reference/validation for cflibs/plasma/partition.py; basis for enforcing one consistent cutoff between U(T) and the Saha Delta-chi. Repo already has a B&C 2016 patch script.
- **Expected benefit:** Partition-function errors propagate linearly into species number densities and composition; using a vetted U(T) with a consistent cutoff removes a multi-percent systematic from the closure equation (polynomial-fit errors up to ~66% for stale fits; ~2% achievable with proper fits). 102 citations.
- **Integration difficulty:** low
- **Citation:** *Partition functions and equilibrium constants for diatomic molecules and atoms of astrophysical interest*, Astronomy & Astrophysics (588, A96; venue blank in S2, DOI + arXiv:1602.03304 confirm A&A 588 2016), 2016 — doi:10.1051/0004-6361/201526961

### Irwin (1981) ln-T polynomial partition-function fits `[DOI-corrected]`
- **Applicability:** forward model (real-time/GPU path): the polynomial U(T) form ln U = sum a_n (ln T)^n already used in cflibs/plasma/partition.py for the vmap'd manifold/Bayesian fallback; confirms the functional form is the literature standard. Fix = refit coefficients vs Barklem & Collet / current NIST levels.
- **Expected benefit:** Sub-2% U(T) error with refit coefficients and O(1) evaluation cost; keeps the GPU/JAX forward model fast without the direct-sum truncation ambiguity. Speed-preserving fidelity. 165 citations.
- **Integration difficulty:** low
- **Citation:** *Polynomial partition function approximations of 344 atomic and molecular species*, Astrophysical Journal Supplement Series (45, 621-633). CORRECTION: survey DOI 10.1086/190731 is a DIFFERENT paper (Vernazza et al., 'Structure of the solar chromosphere III', ApJS 45 1981); the correct Irwin 1981 DOI is 10.1086/190730. Title, author (A.W. Irwin) and ApJS 45:621 page confirmed., 1981 — doi:10.1086/190730

### Ionization-potential depression / Debye-Huckel and Stewart-Pyatt corrections to the Saha equation (Stewart & Pyatt 1966)
- **Applicability:** forward model (Saha balance) + partition cutoff: add a Debye-Huckel/Stewart-Pyatt Delta-chi(T, n_e) to the Saha ionization energy in saha_boltzmann.py and use the SAME Delta-chi as the partition.py truncation cutoff (partition.py already accepts an IPD argument), closing the Saha-U(T) consistency loop.
- **Expected benefit:** Removes the internal inconsistency where the partition sum and Saha ionization energy use different effective ionization limits; corrects ion/neutral ratios at high n_e (few percent at LIBS densities, larger at high-T edge). Required for self-consistency. 545 citations.
- **Integration difficulty:** low
- **Citation:** *Lowering of Ionization Potentials in Plasmas*, Astrophysical Journal (144, 1203-1211; venue blank in S2, DOI confirms ApJ 144 1966), 1966 — doi:10.1086/148714

### CSigma (C-sigma) graphs - generalized curve-of-growth forward model (Aragon & Aguilera 2014)
- **Applicability:** inversion/quantification + self-absorption: an alternative to the Boltzmann-plot + post-hoc self-absorption pipeline that folds optical depth into the forward model; multi-element common-sigma fit aligns with the repo's multi-element common-slope step.
- **Expected benefit:** Demonstrated accurate quantification of geological materials and measurement of transition probabilities; avoids systematic self-absorption error without excluding lines; inhomogeneity control reduces line-of-sight bias. 61 citations.
- **Integration difficulty:** medium
- **Citation:** *CSigma graphs: A new approach for plasma characterization in laser-induced breakdown spectroscopy*, Journal of Quantitative Spectroscopy and Radiative Transfer (149, 90-102; venue blank in S2, DOI confirms JQSRT 149 2014). NOTE: a corrigendum exists (JQSRT 2015, DOI 10.1016/J.JQSRT.2015.03.001)., 2014 — doi:10.1016/J.JQSRT.2014.07.026

### Multiplet-aware Boltzmann plot extension (Volker & Gornushkin 2023)
- **Applicability:** inversion (temperature determination) + line selection: lets the Boltzmann/CD-SB fit in cflibs/inversion/physics/boltzmann.py use unresolved multiplets correctly (summed multiplet intensity with effective gA and energy) instead of discarding them.
- **Expected benefit:** More usable lines and reduced T bias from blends; since T is the dominant exponential error driver, even modest T-precision gains cascade into composition accuracy. 4 citations (recent).
- **Integration difficulty:** medium
- **Citation:** *Extension of the Boltzmann plot method for multiplet emission lines*, Journal of Quantitative Spectroscopy and Radiative Transfer (310, 108741), 2023 — doi:10.1016/j.jqsrt.2023.108741

### Closed-form radiative-transfer error analysis for CF-LIBS optimization (Maali & Shabanov 2019)
- **Applicability:** uncertainty quantification: a physics-grounded sensitivity/Jacobian framework mapping onto the repo's analytical uncertainty path (Boltzmann covariance) in cflibs/inversion/physics/uncertainty.py; identifies dominant forward-model error contributors.
- **Expected benefit:** Provides analytic sensitivity coefficients to prioritize improvements (confirms T and self-absorption dominate) and to produce defensible composition error bars. 7 citations.
- **Integration difficulty:** medium
- **Citation:** *Error analysis in optimization problems relevant for calibration-free laser-induced breakdown spectroscopy*, Journal of Quantitative Spectroscopy and Radiative Transfer (222-223, 236-246), 2019 — doi:10.1016/J.JQSRT.2018.10.029

### Machine-learning surrogates for plasma T/n_e and line-shape inversion (knowledge only) `[phys✗, DOI-corrected]`
- **Applicability:** real-time path (T/n_e estimation surrogate): could pre-seed the iterative solver or manifold nearest-neighbor lookup, but as a surrogate only.
- **Expected benefit:** Near-instant T/n_e estimates (R~0.95-0.99 reported) for real-time triage; accuracy is dataset-specific and not transferable across instruments without retraining.
- **Integration difficulty:** high
- **Citation:** Machine Learning Prediction of Electron Density and Temperature from Optical Emission Spectroscopy in Nitrogen Plasma (representative; survey's primary cite Phys. Plasmas 31:103302 DOI 10.1063/5.0222090 is UNVERIFIABLE -- 404 in asta and 0 search hits), Coatings (11, 1221). CORRECTION: the survey's primary citation 'Simulation of laser-induced plasma temperature based on machine learning, Phys. Plasmas 31 (2024) 103302, DOI 10.1063/5.0222090' could not be verified in asta (DOI 404, search returns 0); the secondary representative citation (Coatings 2021, DOI 10.3390/coatings11101221) is confirmed and anchors the method., 2021 — doi:10.3390/coatings11101221

### Updated NIST critical review of experimental + semiclassical Stark widths/shifts 2008-2020 (Djurovic, Blagojevic & Konjevic 2023) `[NEW]`
- **Applicability:** forward model (broadening) + n_e diagnostic validation: the newest critically-evaluated, accuracy-graded experimental Stark dataset (1665 lines, 35 elements, 61 species, each entry with an accuracy class AND a paired semiclassical theoretical value). Strict newer/stronger replacement for the survey's Konjevic 2002 review; populate per-line Stark widths/shifts and per-line Stark uncertainties in cflibs/radiation broadening and stark_ne.py.
- **Expected benefit:** Doubles the experimentally-vetted line coverage vs the 2002 review and adds an experiment-vs-theory cross-check, exposing lines where theory-only STARK-B values are biased by tens of percent; directly tightens the n_e error bar feeding the Saha correction. Physics-only (a tabulated reference dataset).
- **Integration difficulty:** low
- **Citation:** Experimental and Semiclassical Stark Widths and Shifts for Spectral Lines of Neutral and Ionized Atoms (A Critical Review of Experimental and Semiclassical Data for the Period 2008 Through 2020), Journal of Physical and Chemical Reference Data, 2023 — doi:10.1063/5.0147933

### MERLIN adaptive LTE radiative-transfer forward model validated on reference steel (Favre, Bultel, Morel, Lesage, Gosse 2024) `[NEW]`
- **Applicability:** forward model + inversion (full-spectrum fitting): a recent, mixture-agnostic uniform-LTE radiative-transfer spectrum synthesizer (Saha-Boltzmann populations + Stark/Doppler broadening + slab radiative transfer) validated end-to-end on a certified Eurofer97 steel in Ar. Extends the Hermann ideal-radiation-source / Gornushkin forward-fit paradigm and is a concrete blueprint for using the repo's JAX SpectrumModel as the inversion kernel over arbitrary elemental mixtures.
- **Expected benefit:** Demonstrates that an adaptive single-zone LTE forward model reproduces a real multi-element alloy spectrum well enough for quantification, folding self-absorption and instrument response into one consistent computation (no resonance-line exclusion); a 2024 reference for full-spectrum CF-LIBS fitting accuracy. Physics-only (radiative-transfer code).
- **Integration difficulty:** high
- **Citation:** *MERLIN, an adaptative LTE radiative transfer model for any mixture: Validation on Eurofer97 in argon atmosphere*, Journal of Quantitative Spectroscopy and Radiative Transfer, 2024 — doi:10.1016/j.jqsrt.2024.109222

### SuperCam-on-Mars LIBS plasma diagnostics for quantification (Manelski, Wiens, Bousquet et al. 2024) `[NEW]`
- **Applicability:** forward model + LTE diagnostics + uncertainty, in the repo's exact deployment context (cflibs/pds SuperCam interface): measures plasma T, n_e and LTE validity for SuperCam spectra under Mars-relevant (low-pressure CO2) conditions and ties them to elemental-abundance quantification error. Directly informs the LTE-validity gate and Saha-Boltzmann assumptions for ChemCam/SuperCam-style standoff data.
- **Expected benefit:** Provides the up-to-date, instrument-matched plasma-parameter ranges and LTE-validity envelope for SuperCam, so the inversion can flag/weight spectra where Saha-Boltzmann LTE is marginal under Martian ambient conditions -- closing a gap the generic-laboratory LTE criteria (Cristoforetti 2010/2013) leave open for planetary standoff LIBS. Physics-only.
- **Integration difficulty:** medium
- **Citation:** *LIBS plasma diagnostics with SuperCam on Mars: Implications for quantification of elemental abundances*, Spectrochimica Acta Part B: Atomic Spectroscopy, 2024 — doi:10.1016/j.sab.2024.107061

### Automated Bayesian high-throughput physics-based estimation of plasma T and n_e (Oliver, Michoski, Langendorf, LaJoie 2024) `[NEW]`
- **Applicability:** inversion (T/n_e estimation) + uncertainty: a physics-forward Bayesian estimator of plasma temperature and electron density directly from emission spectra, automated for high throughput. A physics-only alternative to the ML T/n_e surrogates (the survey's only physics_only_compatible=false method): it places posteriors on T and n_e using the Saha-Boltzmann/line-emission physics rather than a learned regressor, aligning with cflibs/inversion/solve/bayesian.
- **Expected benefit:** Gives calibrated posterior uncertainties on T and n_e (not point estimates) at high throughput, propagating rigorous diagnostic error into the composition uncertainty budget; offers the speed/automation motivation behind ML surrogates while staying inside the physics-only constraint. Physics-only (Bayesian inference over the physics model; no sklearn/torch/jax.nn required).
- **Integration difficulty:** medium
- **Citation:** *Automated Bayesian high-throughput estimation of plasma temperature and density from emission spectroscopy*, Review of Scientific Instruments, 2024 — doi:10.1063/5.0192810

### Numerical study of intensity-error propagation into Boltzmann-plot temperature (John & Anoop 2023) `[NEW]`
- **Applicability:** uncertainty quantification: a recent (2023) numerical study quantifying how line-intensity measurement error maps to Boltzmann-plot temperature error, a companion to Maali & Shabanov 2019 and the Tognoni 2007 intrinsic-error framework. Maps onto cflibs/inversion/physics/uncertainty.py to set realistic T error bars from per-line SNR/intensity uncertainty.
- **Expected benefit:** Provides updated, simulation-grounded sensitivity of T to intensity error (the upstream driver of the dominant exponential T-uncertainty in composition); lets the pipeline derive defensible T uncertainties directly from spectrum SNR rather than empirical guesses. Physics-only (numerical error analysis).
- **Integration difficulty:** low
- **Citation:** *Impact of intensity error on temperature estimation in laser-induced breakdown spectroscopy: a numerical study*, Laser Physics, 2023 — doi:10.1088/1555-6611/ad0cb3

### Temporal/relaxation LTE validity criterion beyond McWhirter (Cristoforetti et al. 2010)
- **Applicability:** forward model + uncertainty: gates whether Saha-Boltzmann LTE is physically applicable; evaluate per spectrum/per gate-window as a validity flag feeding the uncertainty budget. Makes cflibs/plasma/lte_validator.py's temporal relaxation check rigorous.
- **Expected benefit:** Prevents systematic bias from applying Saha-Boltzmann outside its validity domain (McWhirter alone is necessary-not-sufficient). Screening criterion; companion CR study quantifies avoided error at 10-42%. 609 citations (canonical).
- **Integration difficulty:** low
- **Citation:** *Local Thermodynamic Equilibrium in Laser-Induced Breakdown Spectroscopy: Beyond the McWhirter criterion*, Spectrochimica Acta Part B: Atomic Spectroscopy (S2 venue field blank; DOI confirms SAB 65:86-95), 2010 — doi:10.1016/j.sab.2009.11.005

### Zero-dimensional collisional-radiative (CR) model coupled to electron Boltzmann equation (Pietanza et al. 2010)
- **Applicability:** forward model: non-LTE benchmark for Saha-Boltzmann level populations; offline map of LTE-validity envelope and per-level departure-coefficient corrections b_k applied multiplicatively in saha_boltzmann.py.
- **Expected benefit:** Quantifies LTE-vs-CR population deviations ~10% at n_e 1e18-1e19, ~27% at 1e17, ~42% at 1e15 cm^-3; defines the density floor below which Saha-Boltzmann inversion is unsafe. 65 citations.
- **Integration difficulty:** high
- **Citation:** *Kinetic processes for laser induced plasma diagnostic: A collisional-radiative model approach*, Spectrochimica Acta Part B: Atomic Spectroscopy (S2 venue blank; DOI confirms SAB 65:616-626), 2010 — doi:10.1016/j.sab.2010.03.012

### Comprehensive thermodynamic-equilibrium-state framework (pLTE, two-temperature, ionizing/recombining) (Cristoforetti, Tognoni, Gizzi 2013)
- **Applicability:** forward model + line selection: physics to weight/exclude lines whose lower level lies below the pLTE thermalization limit; flag ionizing/recombining conditions where the Saha ionic->neutral mapping is biased. Feeds lte_validator and line_selection.
- **Expected benefit:** Reduces systematic Boltzmann-plot bias by restricting fits to thermalized levels; the pLTE-limit concept is the principled basis for E_lower line-selection cuts. 135 citations.
- **Integration difficulty:** medium
- **Citation:** *Thermodynamic equilibrium states in laser-induced plasmas: From the general case to laser-induced breakdown spectroscopy plasmas*, Spectrochimica Acta Part B: Atomic Spectroscopy (SAB 90:1-22; NOTE an erratum exists: DOI 10.1016/j.sab.2014.06.002, paperId 6579de20c41927eab65a91670aacebc3d83cb257), 2013 — doi:10.1016/j.sab.2013.09.004

### Uniform-LTE 'ideal radiation source' forward model with full spectral-radiance fitting (Hermann et al. 2017)
- **Applicability:** forward model + inversion (full-spectrum fitting): replaces/augments the discrete Boltzmann-plot path with a uniform-slab radiative-transfer spectrum fit; maps onto the existing JAX SpectrumModel as the kernel of a least-squares/Bayesian inversion.
- **Expected benefit:** Few-percent to sub-percent accuracy for majors on alloys/thin films when the uniform-LTE window is respected; removes the resonance-line exclusion problem (self-absorption modeled not corrected). 39 citations.
- **Integration difficulty:** high
- **Citation:** *Ideal radiation source for plasma spectroscopy generated by laser ablation*, Physical Review E (96:053210), 2017 — doi:10.1103/PhysRevE.96.053210

### Blackbody-limit diagnostic for LTE and absolute intensity calibration (Hermann et al. 2018)
- **Applicability:** forward model + uncertainty + intensity calibration: in-spectrum LTE proof (thick line cores -> B_lambda(T)) and an absolute-radiance anchor to retrieve E(lambda); implementable as a check that thick line cores in the forward model do not exceed B_lambda(T).
- **Expected benefit:** Self-calibrates spectral response E(lambda) (attacks the response-rotates-Boltzmann-slope systematic) and gives a cheap LTE validity check from the strongest lines. 20 citations.
- **Integration difficulty:** medium
- **Citation:** *Local thermodynamic equilibrium in a laser-induced plasma evidenced by blackbody radiation*, Spectrochimica Acta Part B: Atomic Spectroscopy (144:82-86), 2018 — doi:10.1016/j.sab.2018.03.013

### Computer-simulated (ion-dynamics) Stark line-shape tables for hydrogen Balmer lines (Gigosos, Gonzalez, Cardenoso 2003)
- **Applicability:** electron-density diagnostic + forward model: reference width-n_e relations for the H-Balmer n_e channel; provides ion-dynamics-corrected nonlinear width-n_e tables to replace the linear convention and fix the ~n_e^0.7 H-alpha exponent flagged in stark_ne.py.
- **Expected benefit:** Ion-dynamics-corrected H-beta gives n_e to ~5-10%; correcting n_e^0.7 vs n_e^1 scaling removes a systematic that propagates through Saha into composition. 474 citations.
- **Integration difficulty:** low
- **Citation:** *Computer simulated Balmer-alpha, -beta and -gamma Stark line profiles for non-equilibrium plasmas diagnostics*, Spectrochimica Acta Part B: Atomic Spectroscopy (58:1489-1504), 2003 — doi:10.1016/S0584-8547(03)00097-1

### STARK-B (VAMDC) semiclassical-perturbation Stark width/shift database for non-hydrogenic lines (Sahal-Brechot et al. 2015)
- **Applicability:** forward model (line broadening) + electron-density diagnostic: populate the Stark-coefficient table consumed by cflibs/radiation broadening and stark_ne.py for non-H diagnostic lines (Ca II H/K, Mg II); provides T-dependence of widths.
- **Expected benefit:** Replaces ad hoc/single-value Stark coefficients with theory-grade T-scaled widths for hundreds of lines; improves forward line shapes and Stark-width n_e when H lines are weak/absent. 53 citations.
- **Integration difficulty:** medium
- **Citation:** *The STARK-B database VAMDC node: a repository for spectral line broadening and shifts due to collisions with charged particles*, Physica Scripta (90:054008), 2015 — doi:10.1088/0031-8949/90/5/054008

### NIST critically-evaluated experimental Stark widths and shifts (Konjevic et al. 2002, period 1989-2000) `[DOI-corrected]`
- **Applicability:** forward model (broadening) + n_e diagnostic validation: graded experimental reference to validate/weight Stark coefficients used in broadening and stark_ne.py; assign per-line Stark-coefficient uncertainties propagated into n_e uncertainty.
- **Expected benefit:** Enables accuracy-weighted Stark coefficients and realistic n_e error bars; flags lines where theory-vs-experiment Stark disagreement (tens of %) would silently bias n_e. CORRECTION: survey cited DOI 10.1063/1.1486456 which is NOT this paper (404 in S2, belongs to a different JPCRD article); the correct DOI for this title is 10.1063/1.1525443. 542 citations.
- **Integration difficulty:** low
- **Citation:** Experimental Stark Widths and Shifts for Spectral Lines of Neutral and Ionized Atoms (A Critical Review of Selected Data for the Period 1989 through 2000), Journal of Physical and Chemical Reference Data (31:819-927; S2 year metadata erroneously shows 1990, DOI/title authoritative), 2002 — doi:10.1063/1.1525443

### Self-consistent cutoff-aware partition functions (Barklem & Collet 2016)
- **Applicability:** forward model (Saha-Boltzmann normalization): reference/validation for cflibs/plasma/partition.py; basis for enforcing one consistent IPD cutoff between U(T) and the Saha Delta-chi. Repo already has a B&C 2016 patch script.
- **Expected benefit:** Partition-function errors propagate linearly into species number densities; vetted U(T) + consistent cutoff removes multi-percent systematics from the closure equation (polynomial-fit errors up to ~66% for stale species). 102 citations.
- **Integration difficulty:** low
- **Citation:** *Partition functions and equilibrium constants for diatomic molecules and atoms of astrophysical interest*, Astronomy & Astrophysics (588:A96; arXiv:1602.03304), 2016 — doi:10.1051/0004-6361/201526961

### Irwin (1981) ln-T polynomial partition-function fits `[DOI-corrected]`
- **Applicability:** forward model (real-time/GPU path): the polynomial U(T) form (ln U = sum a_n (ln T)^n) used in cflibs/plasma/partition.py for the vmap'd manifold/Bayesian fallback; refit coefficients against Barklem & Collet / current NIST levels.
- **Expected benefit:** Sub-2% U(T) error with refit coefficients and O(1) evaluation cost; keeps the GPU/JAX forward model fast without direct-sum truncation ambiguity. CORRECTION: survey cited DOI 10.1086/190731 which actually belongs to a different paper ('Structure of the solar chromosphere III'); the correct Irwin DOI is 10.1086/190730. 165 citations.
- **Integration difficulty:** low
- **Citation:** *Polynomial partition function approximations of 344 atomic and molecular species*, Astrophysical Journal Supplement Series (45:621-633), 1981 — doi:10.1086/190730

### Ionization-potential depression / Debye-Huckel and Stewart-Pyatt corrections to the Saha equation (Stewart & Pyatt 1966)
- **Applicability:** forward model (Saha balance) + partition cutoff: add Debye-Huckel/Stewart-Pyatt Delta-chi(T,n_e) to the Saha ionization energy in saha_boltzmann.py and use the SAME Delta-chi as the partition.py truncation cutoff; closes the Saha-U(T) consistency loop.
- **Expected benefit:** Removes internal inconsistency between partition sum and Saha ionization energy; corrects ion/neutral ratios at high n_e (few percent at LIBS densities, larger near the high-T edge); required for self-consistency. 545 citations.
- **Integration difficulty:** low
- **Citation:** *Lowering of Ionization Potentials in Plasmas*, Astrophysical Journal (144:1203-1211), 1966 — doi:10.1086/148714

### CSigma (C-sigma) graphs - generalized curve-of-growth forward model (Aragon & Aguilera 2014)
- **Applicability:** inversion/quantification + self-absorption: alternative to Boltzmann-plot + post-hoc self-absorption that folds optical depth into the forward model; multi-element common-sigma fit aligns with the repo's multi-element common-slope step.
- **Expected benefit:** Accurate quantification of geological materials and even A_ki measurement; avoids self-absorption error without excluding lines; inhomogeneity control reduces line-of-sight bias. 61 citations.
- **Integration difficulty:** medium
- **Citation:** *CSigma graphs: A new approach for plasma characterization in laser-induced breakdown spectroscopy*, Journal of Quantitative Spectroscopy and Radiative Transfer (149:90-102; NOTE corrigendum DOI 10.1016/j.jqsrt.2015.03.001, paperId 3a9398e6b073356a215dc235a61183ad1f293e41), 2014 — doi:10.1016/j.jqsrt.2014.07.026

### Multiplet-aware Boltzmann plot extension (Volker & Gornushkin 2023)
- **Applicability:** inversion (temperature determination) + line selection: lets the Boltzmann/CD-SB fit in cflibs/inversion/physics/boltzmann.py use unresolved multiplets correctly (summed intensity, effective gA and energy) instead of discarding them.
- **Expected benefit:** More usable lines and reduced T bias from blends; since T error propagates exponentially, modest T-precision gains cascade into composition accuracy. 4 citations (recent).
- **Integration difficulty:** medium
- **Citation:** *Extension of the Boltzmann plot method for multiplet emission lines*, Journal of Quantitative Spectroscopy and Radiative Transfer (310:108741), 2023 — doi:10.1016/j.jqsrt.2023.108741

### Closed-form radiative-transfer error analysis for CF-LIBS optimization (Maali & Shabanov 2019)
- **Applicability:** uncertainty quantification: physics-grounded sensitivity/Jacobian framework mapping onto the repo's analytical uncertainty path (Boltzmann covariance) in cflibs/inversion/physics/uncertainty.py.
- **Expected benefit:** Provides analytic sensitivity coefficients to prioritize improvements (confirms T and self-absorption dominate) and to produce defensible composition error bars. 7 citations.
- **Integration difficulty:** medium
- **Citation:** *Error analysis in optimization problems relevant for calibration-free laser-induced breakdown spectroscopy*, Journal of Quantitative Spectroscopy and Radiative Transfer (222-223:236-246), 2019 — doi:10.1016/j.jqsrt.2018.10.029

### Machine-learning surrogates for plasma T/n_e and line-shape inversion (knowledge only; NOT shippable) `[phys✗]`
- **Applicability:** real-time path (T/n_e estimation surrogate): could pre-seed the iterative solver or manifold nearest-neighbor lookup, but as a surrogate only.
- **Expected benefit:** Near-instant T/n_e estimates (R~0.95-0.99) for real-time triage; accuracy is dataset-specific, not transferable across instruments without retraining.
- **Integration difficulty:** high
- **Citation:** *Simulation of laser-induced plasma temperature based on machine learning*, Physics of Plasmas (31:103302). NOTE: not indexed in asta/Semantic Scholar; verified via web fallback (AIP Publishing + NASA ADS 2024PhPl...31j3302W). GPR R~0.99 detail confirmed. Secondary cite (Coatings 11:1221, DOI 10.3390/coatings11101221, paperId e7505795998c780fa5d0fc5c3d0967fa625a857f) is asta-confirmed., 2024 — doi:10.1063/5.0222090

### MERLIN: adaptive LTE radiative-transfer forward model for arbitrary mixtures (Favre, Bultel et al. 2024) `[NEW]`
- **Applicability:** forward model + inversion (full-spectrum fitting): a recent (2024) self-contained uniform-LTE radiative-transfer spectrum synthesizer for ANY element mixture, validated experimentally on the Eurofer97 steel reference in argon. Directly successor to the Hermann 2017 'ideal radiation source' paradigm and a concrete template for using cflibs/radiation/SpectrumModel as a full-spectrum inversion kernel (Saha-Boltzmann populations + Stark/Doppler + slab radiative transfer over T, n_e, composition) instead of discrete Boltzmann-plot intensities.
- **Expected benefit:** Demonstrates an adaptive (any-mixture) LTE RT model validated against a certified steel, i.e. a modern, reproducible blueprint for full-spectrum CF-LIBS that folds self-absorption + instrument response + partition functions into one forward computation, eliminating resonance-line exclusion and post-hoc corrections. Newer and more general than Hermann 2017 (any mixture, open validation).
- **Integration difficulty:** high
- **Citation:** *MERLIN, an adaptative LTE radiative transfer model for any mixture: Validation on Eurofer97 in argon atmosphere*, Journal of Quantitative Spectroscopy and Radiative Transfer (vol. 330, 109222), 2024 — doi:10.1016/j.jqsrt.2024.109222

### Comprehensive collisional-radiative (CR) model for laser-produced Al plasma with ab-initio EIE cross-sections (Ghosh, Baburaj, Thomas, Sharma 2025) `[NEW]`
- **Applicability:** forward model (non-LTE benchmark) + electron-density/temperature diagnostics: a 2025 CR model over 44 fine-structure Al II levels with GRASP2018 (MCDHF) radiative rates and relativistic-distorted-wave electron-impact-excitation cross-sections to ~600 eV, coupled to ns-LIBS Al II line measurements. Modern successor to Pietanza 2010 for benchmarking the shipped Saha-Boltzmann populations and deriving per-level departure coefficients; explicitly compares CR-derived T against Boltzmann-plot T.
- **Expected benefit:** Provides a current, atomic-data-rich non-LTE reference (with vetted EIE cross-sections) to quantify departures of LTE Boltzmann-plot T from true CR-consistent T at LIBS conditions; the GRASP2018/RDW data pipeline is the modern standard for non-hydrogenic CR inputs and improves on the older approximate cross-sections in Pietanza-era models.
- **Integration difficulty:** high
- **Citation:** *A comprehensive collisional radiative model for laser-produced Al plasma*, Plasma Sources Science & Technology, 2025 — doi:10.1088/1361-6595/add9c8

### Self-Calibrated LIBS under partial-LTE with blackbody normalization for volcanic ash (Taleb, Dell'Aglio, De Giacomo et al. 2024) `[NEW]`
- **Applicability:** forward model + inversion under non-LTE / time-integrated spectra: a 2024 partial-LTE (pLTE) self-calibrated method normalizing to the measured blackbody-like continuum so composition is obtained WITHOUT invoking the 100% closure scale factor. Directly relevant to ChemCam/SuperCam-style broadband, time-integrated data where full LTE and clean closure are unsafe; operationalizes the Hermann 2018 blackbody-limit idea for quantification.
- **Expected benefit:** Trueness 'better than 5 wt% on major element concentration' on meteorite/ash matrices at long (5000 ns) gates; value is robustness to non-LTE and time-integration bias plus removal of closure-induced trace-element error, complementing the LTE-gate work. (Cross-listed in topic 1 survey; included here as the recent pLTE forward-model variant.)
- **Integration difficulty:** high
- **Citation:** *Self-Calibrated Laser-Induced Breakdown Spectroscopy for the Quantitative Elemental Analysis of Suspended Volcanic Ash*, Applied Spectroscopy (78:8). Surfaced via forward-citation walk of Hermann 2017; not separately fetched by paperId (asta paperId not captured), DOI from survey cross-checked, citationCount 2., 2024 — doi:10.1177/00037028241241076

## Automated line identification & spectral matching for LIBS/atomic spectra — state of the art (physics-based matching, multi-line intensity coherence, Bayesian/BIC model selection for element presence, cross-correlation/comb methods, NNLS unmixing, and ML classifiers)

### Model–experiment correlation line identification (Saha-Boltzmann simulated spectrum over (T, n_e) grid, maximize Pearson correlation, then attribute peaks)
- **Applicability:** Identification stage tightly coupled to forward model: reuse the existing Saha-Boltzmann synthetic-spectrum generator (cflibs/radiation + cflibs/plasma) to score candidate (T, n_e) by correlation, yielding identification AND a coarse plasma-parameter prior to seed inversion. Maps onto cflibs/inversion/identify (correlation matcher) feeding cflibs/inversion/solve.
- **Expected benefit:** Best correlation 0.943 at T=0.675 eV, lg(n_e)=16.7 on high-alloy steel; >40 lines auto-labeled in 393.34-413.04 nm with no manual intervention. Relative-intensity coherence rejects coincidental single-line false positives.
- **Integration difficulty:** low
- **Citation:** *Automatic identification of emission lines in laser-induced plasma by correlation of model and experimental spectra.*, Analytical Chemistry, 2013 — doi:10.1021/ac303270q

### Comb-like element-specific matched filters correlated with the spectrum (with micro-parameter adjustment for calibration drift) — Gajarska 2024; Amato 2010 text-retrieval precursor `[DOI-corrected]`
- **Applicability:** Identification stage — the comb-matching path already in cflibs/inversion/identify. 2024 micro-parameter adjustment + interference-flagging refinements harden the comb detector against wavelength-calibration error and pre-flag blended lines before cflibs/inversion/physics line_selection.
- **Expected benefit:** Maintains correct element detection under calibration deviations (instrumental fluctuation/drift) that break naive nearest-line matching; autonomously delimits interference regions to keep contaminated lines out of Boltzmann fits.
- **Integration difficulty:** low
- **Citation:** *Automated detection of element-specific features in LIBS spectra*, Journal of Analytical Atomic Spectrometry, 2024 — doi:10.1039/d4ja00247d

### ALIAS — multi-coefficient match of acquired vs. simplified-plasma-model spectrum for fully automated line/element identification in LIBS imaging
- **Applicability:** Identification stage, real-time/high-throughput path. Per-pixel automated identification at imaging scale matches the streaming-inversion workload in cflibs/inversion/runtime; the multi-coefficient scoring is a candidate upgrade to the existing ALIAS-style scorer in cflibs/inversion/identify (which already has 'alias' presets).
- **Expected benefit:** Demonstrated high automation/robustness on real LIBS-imaging datasets (millions of spectra); targets the throughput/automation bottleneck. Most directly relevant 2025 physics-based SOTA for this topic.
- **Integration difficulty:** medium
- **Citation:** *Automated line identification for atomic spectroscopy (ALIAS): Application to LIBS imaging data processing*, Spectrochimica Acta Part B: Atomic Spectroscopy, 2025 — doi:10.1016/j.sab.2025.107255

### Information-criterion model selection for spectra (SpIC vs AICc vs BIC) to decide how many real lines/components are present
- **Applicability:** Identification + uncertainty: principled element-presence / line-reality decision. Replaces ad-hoc detection thresholds in cflibs/inversion/identify (BIC model selection) with a calibrated penalized-likelihood test; the weak-vs-strong-line penalty suits LIBS where many candidate elements appear only via faint lines, and informs how many lines per element enter the Boltzmann fit (cflibs/inversion/physics).
- **Expected benefit:** More-correct component counts than AICc/BIC: fewer parameters than AICc at equal accuracy, avoids BIC underfitting; reduces both false-positive element calls and missed faint elements. Calibrated Occam decision rule replacing a hand-tuned correlation cutoff.
- **Integration difficulty:** medium
- **Citation:** *Getting the model right: an information criterion for spectroscopy*, Monthly Notices of the Royal Astronomical Society (arXiv:2009.08336), 2020 — doi:10.1093/mnras/staa3551

### Bayesian evidence / Bayes-factor model selection for element presence (variational line-spectra, Jeffreys-scale thresholds) — Badiu 2017; Lewis ChemCam 2020 `[DOI-corrected]`
- **Applicability:** Identification + UQ: produces a posterior probability of element presence (not yes/no), the 'element presence with uncertainty' deliverable. Fits cflibs/inversion/identify (BIC model selection) and cflibs/inversion/solve/bayesian (NumPyro/dynesty); the dynesty path already computes evidence, so Bayes-factor element-presence selection is near-term reuse. Variational line-spectra implementable in numpy/jax.numpy.
- **Expected benefit:** Calibrated per-element presence probabilities and principled multiplicity (number-of-components) selection; Jeffreys-scale gives interpretable confidence. ChemCam Bayesian-calibration work shows explicit model-discrepancy modeling improves uncertainty realism on real planetary LIBS.
- **Integration difficulty:** medium
- **Citation:** *Variational Bayesian Inference of Line Spectra*, IEEE Transactions on Signal Processing 65(9) (arXiv:1604.03744), 2017 — doi:10.1109/TSP.2017.2655489

### Non-negative least squares (NNLS) spectral unmixing against a pure-element/reference-spectrum library — Lawson-Hanson algorithm; Miller 2018 detection-ambiguity demo
- **Applicability:** Identification + quantification: NNLS unmixing is the spectral-NNLS / hybrid (NNLS+ALIAS) identifier and candidate_prefilter (select_candidate_elements is NNLS top-K). Sparse/L1-regularized NNLS + library-coherence handling harden cflibs/inversion/identify/spectral and cflibs/inversion/candidate_prefilter against false positives and give a fast linear quantification seed for the iterative loop.
- **Expected benefit:** Sparse non-negative abundance vector = simultaneous presence detection + coarse quantification in one convex deterministic solve (scipy.optimize.nnls); demonstrated to 'reduce detection ambiguity' for in-situ space spectroscopy. Cheap (one active-set solve) — good real-time prefilter before expensive inversion.
- **Integration difficulty:** low
- **Citation:** Fitting Cometary Sampling and Composition Mass Spectral Results Using Non-negative Least Squares: Reducing Detection Ambiguity for In Situ Solar System Organic Compound Measurements, ACS Earth and Space Chemistry, 2018 — doi:10.1021/acsearthspacechem.8b00122

### 1D/2D Convolutional Neural Network classifiers for LIBS spectra (lithology / material / element-class) `[phys✗]`
- **Applicability:** Identification stage — FRONTIER/KNOWLEDGE ONLY (CANNOT ship; physics-only constraint, no torch/tf/jax.nn). Maps to material/class ID and element-presence screening; informs which line groups carry discriminative power for physics filters. Useful as an offline benchmark ceiling.
- **Expected benefit:** Validation/test accuracy 0.9877 / 1.00 on rock-type classification (dolomite, granite, limestone, mudstone, shale), beating 1D-CNN, kNN, PCA-kNN, SVM, PCA-SVM, PLS-DA and human-assisted ANN baselines. Establishes the empirical upper bound for physics identifiers.
- **Integration difficulty:** high
- **Citation:** Convolutional neural network as a novel classification approach for laser-induced breakdown spectroscopy applications in lithological recognition, Spectrochimica Acta Part B: Atomic Spectroscopy, 2020 — doi:10.1016/j.sab.2020.105801

### Distance/matrix-robust deep CNN with sample-weight optimization for multi-distance LIBS (planetary, MarSCoDe/Tianwen-1) `[phys✗]`
- **Applicability:** Identification, real-time/standoff path — KNOWLEDGE ONLY (physics_only_compatible=false). Relevant because the repo carries SuperCam/ChemCam (cflibs/pds) and MarSCoDe data; the concept of explicitly modeling distance/matrix-induced spectral variation when scoring identity ports to physics methods by widening (T, n_e)/intensity priors of comb/correlation matchers across standoff.
- **Expected benefit:** 92.06% test accuracy on 8-distance MarSCoDe-replica data (37 CRMs, 6 classes, 17,760 spectra); +8.45 pts accuracy, +6.4 precision, +7.0 recall, +8.2 F1 vs unweighted CNN; beats BPNN/SVM/LDA/LR. Quantifies the value of distance/matrix robustness for presence classification.
- **Integration difficulty:** high
- **Citation:** A multi-distance laser-induced breakdown spectroscopy data classification method based on deep convolutional neural network and spectral sample weight optimization, Scientific Reports, 2025 — doi:10.1038/s41598-025-24644-x

### Transformer / self-attention classifiers for 1D analytical spectra (SpectraTr) `[phys✗]`
- **Applicability:** Identification stage — FRONTIER/KNOWLEDGE ONLY (physics_only_compatible=false). Attention over wavelength channels models multi-line coherence — the same physical signal comb/correlation methods exploit by hand; argues for encoding long-range, all-pairs line co-occurrence in physics filters. Offline benchmark, not shippable.
- **Expected benefit:** Up to 100% (train) / 99.52% (test) at 8:2 split; 96.97% with NO preprocessing on a public drug dataset — +34.85/+28.28/+5.05/+2.73 pts over PLS-DA/SVM/SAE/CNN. Marks the ML ceiling for spectral qualitative analysis.
- **Integration difficulty:** high
- **Citation:** *SpectraTr: A novel deep learning model for qualitative analysis of drug spectroscopy based on transformer structure*, Journal of Innovative Optical Health Sciences, 2022 — doi:10.1142/s1793545822500213

### Full-spectrum Voigt decomposition with shoulder detection + residual completion for LIBS spectral interference / overlapped features `[NEW]`
- **Applicability:** Identification + line-selection: a purely physics-based deconvolution that resolves dense line crowding, plasma broadening, and overlapping features into individual Voigt components. Directly upgrades the interference-flagging/blended-line handling in cflibs/inversion/physics line_selection and the comb-detector's interference-region delimitation, deciding which overlapped peaks are real lines before they enter Boltzmann fits. Found via forward-citation walk on Labutin 2013.
- **Expected benefit:** Recovers interfered/overlapped lines (shoulder detection + residual completion) that naive nearest-line matching loses, improving both identification completeness in crowded windows and downstream quantitative performance. Newest (2026) physics-based answer to the same interference problem Gajarska 2024 flags qualitatively.
- **Integration difficulty:** medium
- **Citation:** From Spectral Interference to Overlapped Features: A Full-Spectrum LIBS Voigt Decomposition Strategy with Shoulder Detection and Residual Completion., Analytical Chemistry, 2026 — doi:10.1021/acs.analchem.5c07415

### Element recognition by comparing vectors of peak quantities (physics/feature-vector matching of detected-peak attributes against per-element reference vectors) `[NEW]`
- **Applicability:** Identification stage: a deterministic, library-driven element-recognition scheme that compares vectors of detected peak quantities to per-element reference vectors — an alternative/complement to comb and correlation matchers in cflibs/inversion/identify, and a cheap scoring layer that does not require a full forward-model fit. Found via forward-citation walk on Labutin 2013.
- **Expected benefit:** Lightweight, interpretable element presence scoring based on observed peak-quantity vectors (no ML model needed); recent (2024) and same-venue as ALIAS/comb work, providing an independent physics-based identifier to ensemble against the existing comb/correlation/NNLS paths to cut false positives.
- **Integration difficulty:** low
- **Citation:** *Element recognition of laser-induced breakdown spectroscopy by comparing vectors of peak quantities*, Spectrochimica Acta Part B: Atomic Spectroscopy, 2024 — doi:10.1016/j.sab.2024.106927

### Group-sparse variational line-spectra inference (variational EM, infers number of line-groups AND lines-per-group via group-sparsity) — direct successor to Badiu 2017 `[NEW]`
- **Applicability:** Identification + uncertainty: extends the cited Badiu 2017 variational line-spectra estimator. Each element's expected lines form a 'group' tied to a common parameter (the element / plasma state); the variational-EM group-sparse solution infers BOTH which element-groups are present AND how many lines per group on a continuum — i.e. element-presence and per-element line multiplicity in one Occam-penalized inference. Maps onto cflibs/inversion/identify (BIC model selection) as a more structured replacement; implementable in numpy/jax.numpy (no ML libraries).
- **Expected benefit:** Turns 'which elements + how many of their lines are real' into a single group-sparse posterior with automatic complexity control, giving calibrated presence + multiplicity decisions that respect the element-grouping structure LIBS line lists naturally have — stronger than per-line BIC and than ungrouped variational line spectra. Found via forward-citation/related search on Badiu 2017.
- **Integration difficulty:** medium
- **Citation:** *Variational Inference of Structured Line Spectra Exploiting Group-Sparsity*, IEEE Transactions on Signal Processing (arXiv:2303.03017), 2023 — doi:10.1109/TSP.2024.3493603

### Sparse-perspective hyperspectral pixel unmixing with large spectral libraries (modern theory + practice for sparse-NNLS against big correlated element libraries) `[NEW]`
- **Applicability:** Identification + prefilter: the 2024 sparse-unmixing theory backing the NNLS/sparse-NNLS direction in cflibs/inversion/candidate_prefilter (select_candidate_elements) and cflibs/inversion/identify/spectral. Provides identifiability/recovery conditions and practical algorithms for unmixing against large, correlated pure-element libraries — exactly the regime when many candidate elements share overlapping lines.
- **Expected benefit:** Library-coherence-aware sparse unmixing reduces false detections when the element library is large and correlated (the LIBS failure mode), hardening the cheap convex NNLS prefilter that gates the expensive Bayesian/iterative inversion. Convex/deterministic, scipy-implementable; no ML libraries. (Already referenced as a supplementary cite under the NNLS method; promoted here as a standalone 2024 SOTA refinement.)
- **Integration difficulty:** medium
- **Citation:** *Theoretical and Practical Progress in Hyperspectral Pixel Unmixing with Large Spectral Libraries from a Sparse Perspective*, IEEE WHISPERS (Workshop on Hyperspectral Image and Signal Processing); arXiv:2408.07580, 2024 — doi:10.1109/WHISPERS65427.2024.10876427

### Self-training automated spectral-line identification code for plasma spectra ('Whose line is it anyway?') `[NEW]`
- **Applicability:** Identification stage: an automated, self-training line-ID scheme for plasma spectra that bootstraps line assignments from physical line lists — a cross-domain (fusion/plasma) analogue to LIBS comb/correlation identification. Concept transfers to cflibs/inversion/identify as a self-consistent assignment loop that iteratively refines which atomic lines explain the observed peaks. NOTE: 'self-training' here is iterative physical bootstrapping, not necessarily a banned ML library — verify the implementation before any code reuse; included as a frontier concept, not a drop-in.
- **Expected benefit:** Automated, iteratively-refined line attribution reduces manual assignment effort and improves consistency on crowded plasma spectra; a complementary identification paradigm to forward-model correlation (Labutin) and comb matching. Borderline on physics-only depending on the underlying classifier — flagged for verification.
- **Integration difficulty:** high
- **Citation:** *Whose line is it anyway? A self-training spectral line identification code for plasma physics experiments*, (plasma-physics line-ID code; surfaced via forward-citation walk on Labutin 2013), 2022

## Spectrum→composition inverse methods from adjacent fields (stellar abundances, XRF/XRD fundamental parameters, atmospheric/exoplanet retrieval, chemometrics) transferable to CF-LIBS quantification

### Full-spectrum synthetic fitting (stellar spectral synthesis) replacing/augmenting Boltzmann-plot equivalent-width analysis `[DOI-corrected]`
- **Applicability:** inversion-quantification + forward model: the synthesis branch is exactly the project's Hermann-style full-spectrum forward fit; the EW branch is the Boltzmann/Saha solver but disciplined by jointly enforcing excitation-equilibrium AND ionization-balance as coupled constraints, directly hardening the IterativeCFLIBSSolver T/ne loop.
- **Expected benefit:** robustness/accuracy: synthesis fitting recovers usable signal from blended and partly self-absorbed lines that EW/Boltzmann methods discard; the joint excitation+ionization constraint removes the single-Boltzmann-plot degeneracy that lets T drift. iSpec/SME report ~50–100 K Teff, ~0.1 dex logg, ~0.05–0.1 dex (~12–25% relative) abundance precision on high-S/N spectra — the realistic floor of this physics-only inverse class.
- **Integration difficulty:** medium
- **Citation:** *Determining stellar atmospheric parameters and chemical abundances of FGK stars with iSpec*, Astronomy & Astrophysics (A&A 569, A111), 2014 — doi:10.1051/0004-6361/201423945

### Curve-of-growth + microturbulence formalism for the saturated/self-absorbed line regime
- **Applicability:** self-absorption: per-line trust weighting (which COG branch — linear/flat/damping) plus a self-consistency criterion (concentration independent of line strength) to fit the self-absorption nuisance parameter from data alone, complementing CDSB/IRSAC in inversion/physics.
- **Expected benefit:** accuracy/robustness: replaces brittle resonance-line exclusion with continuous COG-branch weighting; the 'abundance independent of reduced EW' criterion is the stellar gold-standard self-consistency test, directly portable as a self-absorption validity check.
- **Integration difficulty:** medium
- **Citation:** *A procedure for correcting self-absorption in calibration free-laser induced breakdown spectroscopy (Bulajic et al.)*, Spectrochimica Acta Part B (57, 339), 2002 — doi:10.1016/S0584-8547(01)00398-6

### Optimal Estimation (Rodgers MAP) — regularized Bayesian inversion with averaging kernels, error covariance, and degrees-of-freedom-for-signal
- **Applicability:** inversion-quantification + uncertainty: J(x) minimization is the project's joint L-BFGS-B optimizer plus a Gaussian-prior regularizer; posterior covariance S_hat augments the analytical uncertainty module; averaging kernels + DOF give per-element identifiability. Implementable with numpy/scipy/jax.numpy (jacobians via jax autodiff of the existing forward model) — no jax.nn.
- **Expected benefit:** uncertainty + robustness: well-posed regularized problem with closed-form correlation-aware covariance (no Monte Carlo re-runs for first-order UQ); DOF metric objectively flags spectrum-determined vs prior-driven concentrations — directly addresses the closure-error-corrupts-low-abundance finding.
- **Integration difficulty:** medium
- **Citation:** *Inverse Methods for Atmospheric Sounding: Theory and Practice (Rodgers)*, World Scientific (Series on Atmospheric, Oceanic and Planetary Physics, Vol. 2), 2000 — doi:10.1142/3171

### Nested sampling / Bayesian-evidence model selection (exoplanet & planetary atmospheric retrieval) `[DOI-corrected]`
- **Applicability:** identification + inversion-quantification + uncertainty: Bayes-factor element-presence detection maps onto the identify/ stage as a physics-grounded model-selection gate; evidence-based forward-model selection picks single- vs two-zone in solve/; nested sampling hardens the existing dynesty solver against multimodality.
- **Expected benefit:** robustness + correct UQ: nested sampling returns the full (possibly multimodal) posterior and evidence Z in one run; Bayes factors give an objective element-detection criterion (|Δ ln Z| > 5 = 'strong'). dynesty is pure-Python/numpy — physics-only.
- **Integration difficulty:** low
- **Citation:** *Atmospheric Retrieval of Exoplanets (Madhusudhan, Handbook of Exoplanets)*, Handbook of Exoplanets, Springer (arXiv:1808.04824), 2018 — doi:10.1007/978-3-319-55333-7_104

### X-ray fluorescence Fundamental-Parameters (Sherman equation) iterative standardless quantification with explicit closure
- **Applicability:** inversion-quantification + self-absorption analogue: the matrix-coupled fixed-point-with-closure (Sum C = 1) iteration is a hardened template for the IterativeCFLIBSSolver closure loop; XRF secondary-enhancement is the conceptual analogue of LIBS radiative-transfer self-absorption.
- **Expected benefit:** accuracy/robustness: FP standardless XRF reaches ~few-percent-relative on majors with good atomic data and degrades predictably with atomic-data quality; influence-coefficient (Lachance-Traill/COLA) updates converge in ~2–4 iterations vs full-FP recompute, stabilizing/accelerating the closure loop.
- **Integration difficulty:** medium
- **Citation:** *The theoretical derivation of fluorescent X-ray intensities from mixtures (Sherman)*, Spectrochimica Acta (7, 283–306), 1955 — doi:10.1016/0371-1951(55)80041-0

### MCR-ALS (Multivariate Curve Resolution – Alternating Least Squares) with non-negativity, closure, and equality/physical constraints
- **Applicability:** identification + inversion-quantification: constrained-ALS spectral unmixing strengthens the spectral-NNLS identifier in identify/; closure-as-projection hardens the solve/ closure step; hard+soft hybrid modelling is a physics-respecting data-fusion route; rotational-ambiguity bands (MCR-BANDS) give a composition identifiability diagnostic.
- **Expected benefit:** robustness: non-negativity + closure + selectivity constraints regularize unmixing of overlapping multi-element windows that defeat peak-by-peak identification; MCR-BANDS quantifies solution non-uniqueness. Implementable entirely with scipy.optimize.nnls / numpy least squares — physics-only.
- **Integration difficulty:** medium
- **Citation:** *Multivariate Curve Resolution: 50 years addressing the mixture analysis problem - A review (de Juan & Tauler)*, Analytica Chimica Acta (1145, 59–78), 2021 — doi:10.1016/j.aca.2020.10.051

### Critically-evaluated, quality-graded line-list infrastructure (VALD3) with damping parameters for the forward model `[DOI-corrected]`
- **Applicability:** forward model: per-line Stark/vdW damping constants and HFS feed radiation/profiles.py for physically correct Voigt profiles and a per-line (not bulk) Stark ne diagnostic; the graded-provenance discipline extends the existing NIST A_ki grading to broadening data.
- **Expected benefit:** accuracy: correct per-line damping removes a systematic profile error biasing both line-integration intensities and full-spectrum chi^2; HFS removes a known overbroadening/intensity bias on Mn/Cu/Ba/Eu-type lines — the data-quality precondition for the synthesis-fitting gains in method 1.
- **Integration difficulty:** low
- **Citation:** *A major upgrade of the VALD database (Ryabchikova et al.)*, Physica Scripta (90, 054005), 2015 — doi:10.1088/0031-8949/90/5/054005

### Dominant-factor hybrid model: physics forward model as the leading term + multivariate residual correction (LIBS, ported chemometric discipline) `[phys✗, DOI-corrected]`
- **Applicability:** inversion-quantification (architecture) + preprocess (spectrum standardization): physics-as-dominant-term with residual correction is a design pattern for fusing the CF-LIBS forward model with empirical matrix-effect correction; spectrum standardization is a directly portable preprocessing step.
- **Expected benefit:** accuracy: combined spectrum-standardization + dominant-factor PLS reports markedly lower RMSEP for carbon-in-coal than either component alone. NOTE: the PLS correction term uses multivariate regression — keep as offline diagnostic / physics_only_compatible=false for the shipped path; the spectrum-standardization preprocessing IS physics-only.
- **Integration difficulty:** medium
- **Citation:** A model combining spectrum standardization and dominant factor based partial least square method for carbon analysis in coal using laser-induced breakdown spectroscopy (Feng et al.), Spectrochimica Acta Part B (102, 52–57), 2014 — doi:10.1016/j.sab.2014.06.017

### UltraNest — reactive/MLFriends nested sampling as a modern, pure-Python evidence engine (successor to MultiNest) `[NEW]`
- **Applicability:** inversion-quantification + uncertainty + identification: drop-in replacement/complement to the shipped dynesty for the nested-sampling/Bayesian-evidence method (verified method 4). Reactive nested sampling + MLFriends region construction + step samplers handle the multimodal, high-dimensional CF-LIBS T/ne/composition posterior more robustly and return calibrated ln Z for Bayes-factor element-presence and forward-model-complexity selection.
- **Expected benefit:** robustness + correct UQ: more reliable evidence estimates and multimodal exploration than MultiNest with self-tuning live-point management; pure-Python/numpy (cython acceleration optional), no sklearn/torch/tf — a stronger, currently-maintained instantiation of the 2009-era MultiNest the survey cited.
- **Integration difficulty:** low
- **Citation:** *UltraNest - a robust, general purpose Bayesian inference engine (Buchner)*, Journal of Open Source Software (6, 3001), 2021 — doi:10.21105/joss.03001

### Modern nested-sampling methodology review (Buchner 2021) — algorithm-selection guidance for evidence-based CF-LIBS model comparison `[NEW]`
- **Applicability:** inversion-quantification + uncertainty: the up-to-date (post-MultiNest/post-Skilling) methodological reference for choosing region samplers, step samplers, live-point counts and stopping criteria when adopting evidence-based model selection in solve/ — the missing 'how to do it correctly in 2020s' companion to Skilling 2006.
- **Expected benefit:** robustness: codifies failure modes (region under-/over-estimation, biased Z) and remedies for the multimodal LIBS posterior, reducing the risk of mis-calibrated Bayes factors in element-presence gating.
- **Integration difficulty:** low
- **Citation:** *Nested sampling methods (Buchner)*, Statistics Surveys (arXiv:2101.09675), 2021 — doi:10.1214/23-SS144

### Korg — differentiable / gradient-enabled stellar spectral synthesis with built-in least-squares fitting and model-atmosphere interpolation `[NEW]`
- **Applicability:** forward model + inversion-quantification: a 2023 instantiation of the synthesis-fitting branch of verified method 1, but engineered for fast, autodiff-friendly synthesis and direct parameter fitting — the design template for making the CF-LIBS forward model itself differentiable (jax autodiff jacobians) so synthesis chi^2 minimization and Optimal-Estimation jacobians come for free.
- **Expected benefit:** robustness/speed: shows that full-spectrum synthesis fitting (blends, wings, self-absorbed lines) can be made fast and gradient-based with pure numeric code, validating an autodiff CF-LIBS forward fit; physics-only (numeric synthesis, no neural nets).
- **Integration difficulty:** medium
- **Citation:** *Korg: Fitting, Model Atmosphere Interpolation, and Brackett Lines (Wheeler et al.)*, The Astronomical Journal (167, 83), 2023 — doi:10.3847/1538-3881/ad19cc

### POSEIDON — open-source multidimensional atmospheric retrieval code (physical forward model + nested-sampling inference) `[NEW]`
- **Applicability:** inversion-quantification + uncertainty: a concrete, recent, well-documented reference architecture for coupling a parametric physical forward model to nested-sampling Bayesian inference with evidence-based model comparison — a direct blueprint for restructuring solve/ around the verified-method-3/4 pattern, including prior specification and forward-model-complexity selection.
- **Expected benefit:** robustness + correct UQ + engineering: demonstrates the full forward-model + nested-sampling + evidence pipeline in maintained open-source Python (numpy/scipy + sampler), de-risking the CF-LIBS port; physics-only (no ML in the retrieval core).
- **Integration difficulty:** medium
- **Citation:** *POSEIDON: A Multidimensional Atmospheric Retrieval Code for Exoplanet Spectra (MacDonald)*, Journal of Open Source Software (8, 4873), 2023 — doi:10.21105/joss.04873

### archNEMESIS — modern open-source Python rewrite of the NEMESIS Optimal-Estimation retrieval tool `[NEW]`
- **Applicability:** inversion-quantification + uncertainty: the current-generation, Python, openly-licensed implementation of Rodgers Optimal Estimation (verified method 3) with forward-model + Gauss-Newton/LM + posterior-covariance/averaging-kernels/DOF — a directly readable reference for implementing OE on the CF-LIBS forward model in numpy/scipy/jax.
- **Expected benefit:** uncertainty + robustness: provides a maintained, inspectable OE codebase (K-matrix construction, S_hat, A, DOF, retrieval iteration) to port rather than re-derive from the 2000 Rodgers monograph; physics-only.
- **Integration difficulty:** medium
- **Citation:** *archNEMESIS: An Open-Source Python Package for Analysis of Planetary Atmospheric Spectra (Alday et al.)*, Journal of Open Research Software (arXiv:2501.16452), 2025 — doi:10.5334/jors.554

### Catalog of Exoplanet Atmospheric Retrieval Codes — curated landscape of forward-model + Bayesian-inversion pipelines `[NEW]`
- **Applicability:** inversion-quantification + uncertainty (reference/survey): a 2023 census of ~40 spectrum→composition retrieval codes (their forward models, samplers — nested sampling vs MCMC vs SBI — and licensing), letting the CF-LIBS effort pick the closest-matching, physics-only architecture to import rather than reinvent.
- **Expected benefit:** robustness/engineering: shortcuts the build-vs-borrow decision for the inversion backbone by mapping which adjacent-field codes are open, physics-based, and sampler-compatible with the project's dynesty/numpy stack.
- **Integration difficulty:** low
- **Citation:** *A Catalog of Exoplanet Atmospheric Retrieval Codes*, Research Notes of the AAS (7, 54), 2023 — doi:10.3847/2515-5172/acc46a

## State-of-the-art uncertainty quantification for CF-LIBS spectroscopic inversion: trustworthy, calibrated uncertainty on plasma parameters (T, n_e) and elemental composition

### Pragmatic- vs Fully-Bayesian propagation of ATOMIC-DATA systematic uncertainty into plasma parameters (Capella framework) `[DOI-corrected]`
- **Applicability:** uncertainty + inversion-quantification: directly maps to propagating A_ki / partition-function / energy-level uncertainty through the Boltzmann-plot + Saha + closure chain. Supplies the PCA-compressed atomic-data-prior + pragmatic/fully-Bayesian distinction the current MonteCarloUQ lacks (it treats atomic data as independent per-line Gaussians with no cross-line correlation, no spectrum-driven update).
- **Expected benefit:** Recovered T and n_e with honest 0.1 dex (T) / 0.2 dex (n_e) uncertainties; demonstrated atomic-data systematics are comparable to or LARGER than statistical errors, so a noise-only MC understates composition uncertainty. Fully-Bayesian sharpens errors vs pragmatic by letting the spectrum constrain the atomic data.
- **Integration difficulty:** high
- **Citation:** *Effect of Systematic Uncertainties on Density and Temperature Estimates in Coronae of Capella*, The Astrophysical Journal, 2024 — doi:10.3847/1538-4357/ad4108

### Hamiltonian Monte Carlo with the No-U-Turn Sampler (NUTS) for plasma-parameter + composition posteriors
- **Applicability:** uncertainty + inversion-quantification: the engine behind the pipeline's NumPyroNUTSSampler. Gold-standard for full posterior UQ over (T, n_e, composition, nuisance baseline); captures the strong T<->intercept<->concentration correlations the Boltzmann-plot linearization only approximates.
- **Expected benefit:** Effective-sample-size per gradient eval orders of magnitude higher than random-walk Metropolis; full non-Gaussian credible intervals rather than symmetric +/- sigma.
- **Integration difficulty:** low
- **Citation:** *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*, Journal of Machine Learning Research 15:1593-1623, 2014 — doi:10.5555/2627435.2638586

### Dynamic Nested Sampling (dynesty) for evidence + multimodal composition posteriors
- **Applicability:** uncertainty + identification (model selection): backs the pipeline's DynestyNestedSampler. Gives evidence Z for probabilistic BIC-style element/model selection and robust posteriors when the composition posterior is multimodal (line-ID ambiguity, ionization-stage degeneracy) -- a CF-LIBS failure mode the iterative solver cannot express.
- **Expected benefit:** Robust posteriors on multimodal/degenerate problems where MCMC under-samples modes; direct calibrated evidence estimates with quantified statistical+sampling error for model comparison.
- **Integration difficulty:** low
- **Citation:** *dynesty: a dynamic nested sampling package for estimating Bayesian posteriors and evidences*, Monthly Notices of the Royal Astronomical Society 493(3):3132-3158, 2020 — doi:10.1093/mnras/staa278

### Simulation-Based Calibration (SBC) to certify that posterior credible intervals are actually calibrated
- **Applicability:** uncertainty (validation): a self-consistency harness over the existing JAX forward model + NUTS/dynesty samplers. Adds the missing calibration-coverage certificate (rank-uniformity) on top of the repo's GoldenSpectrum round-trip validator and split-R-hat/ESS diagnostics.
- **Expected benefit:** Detects miscalibrated/biased posteriors that R-hat and ESS cannot (those test convergence, not coverage). Yields a publishable rank-uniformity calibration histogram per parameter; pure numpy/scipy (rank computation + chi-square uniformity test).
- **Integration difficulty:** medium
- **Citation:** *Validating Bayesian Inference Algorithms with Simulation-Based Calibration*, arXiv:1804.06788, 2018 — S2:e11817ce34636abb2aedf31442c040aee12208a4

### Conformal Prediction (split CP) for distribution-free, finite-sample coverage-guaranteed composition intervals
- **Applicability:** uncertainty (calibrated intervals): an outer wrapper over the CF-LIBS inversion output, calibrated on certified reference samples (repo has supercam_calib and synthetic_corpus). Converts point composition + Bayesian/MC sigma into intervals with a PROVEN coverage guarantee, hedging against Saha-Boltzmann model misspecification that pure Bayesian intervals do not protect against.
- **Expected benefit:** Finite-sample, distribution-free marginal coverage exactly at the target 1-alpha. Trade-off: marginal not conditional coverage; intervals can be conservatively wide if base uncertainty is poorly shaped.
- **Integration difficulty:** low
- **Citation:** *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, arXiv:2107.07511, 2021 — S2:c3ea8eb80bc8ca0b21efa273b9e4a9fd059c65be

### Conformalized Quantile Regression (CQR) for adaptive, heteroscedastic composition intervals
- **Applicability:** uncertainty (calibrated intervals): same outer-wrapper slot as split CP but heteroscedasticity-aware -- valuable because LIBS composition uncertainty is strongly concentration-dependent (large for trace, small for major). Quantile model can be a quantile-loss linear/spline fit (scipy) over CF-LIBS-derived features, keeping it physics-stack-compatible if no sklearn is used.
- **Expected benefit:** Maintains exact marginal coverage while producing materially shorter intervals than constant-width CP under heteroscedasticity; intervals widen automatically for trace elements and tighten for majors.
- **Integration difficulty:** medium
- **Citation:** *Conformalized Quantile Regression*, Advances in Neural Information Processing Systems (NeurIPS) 32; arXiv:1905.03222, 2019 — S2:6f9dc6f8519e927d948a13aa7ae0df336f443eb9

### Optimal Estimation (Rodgers) retrieval: analytic posterior covariance, averaging kernels and degrees-of-freedom for signal
- **Applicability:** uncertainty + inversion-quantification + real-time: a fast Levenberg-Marquardt path (repo's joint_optimizer.py is L-BFGS-B-based) returning posterior covariance + averaging kernels + DOFS in one linear-algebra step. Calibrated, correlation-aware UQ far faster than MCMC; averaging kernel diagnoses which elements are prior-dominated (untrustworthy) vs data-constrained.
- **Expected benefit:** Full analytic error covariance + smoothing/measurement/parameter error partition + DOFS in essentially one Jacobian evaluation; orders of magnitude faster than sampling; operational standard for satellite trace-gas retrievals.
- **Integration difficulty:** medium
- **Citation:** *Inverse Methods for Atmospheric Sounding: Theory and Practice*, World Scientific (Series on Atmospheric, Oceanic and Planetary Physics, Vol. 2), 2000 — doi:10.1142/3171

### Grid-based Bayesian posterior over (T, n_e) with a Gaussian-process model-discrepancy term `[DOI-corrected]`
- **Applicability:** forward model + inversion-quantification + uncertainty: maps onto the repo's manifold (pre-computed spectra on a (T,n_e,composition) grid) -- the manifold IS the grid, so a grid-posterior is a natural MCMC-free UQ layer on top of coarse_to_fine.py. The GP discrepancy term directly addresses forward-model misspecification (LTE/optically-thin assumptions) that otherwise produces over-confident plasma parameters.
- **Expected benefit:** Robust full marginal posteriors on T, n_e without sampler-convergence risk; the GP discrepancy widened intervals enough to show apparent inter-condition T shifts (~tenths of an eV) were NOT significant -- prevented over-confident false detections. Reported T uncertainty intervals ~0.5 eV wide.
- **Integration difficulty:** medium
- **Citation:** *Automated Bayesian high-throughput estimation of plasma temperature and density from emission spectroscopy*, Review of Scientific Instruments 95(7), 2024 — doi:10.1063/5.0192810

### Neural density estimation (normalizing flows) for amortized, well-calibrated LIBS state posteriors (knowledge / ML-only) `[phys✗]`
- **Applicability:** inversion-quantification + uncertainty + real-time: amortized neural posterior is the fastest possible UQ at inference (single pass), conceptually the ML counterpart to the manifold/coarse-to-fine path. Reported well-calibrated on real ChemCam Mars data.
- **Expected benefit:** Amortized (constant-time) posteriors with stated well-calibrated uncertainties on real Mars ChemCam LIBS; shows the direction the physics-only manifold path could emulate without neural nets (kernel-density / grid posterior over the precomputed manifold).
- **Integration difficulty:** high
- **Citation:** *Neural density estimation and uncertainty quantification for laser induced breakdown spectroscopy spectra*, arXiv:2108.08709 (NeurIPS 2021 ML4PS workshop), 2021 — S2:2161128430b60f81c32f74c778d6c183696f3fbb

### TARP coverage testing (Tests of Accuracy with Random Points) -- evaluation-free posterior calibration certificate `[NEW]`
- **Applicability:** uncertainty (validation): a stronger, newer complement/alternative to SBC (method 3) for certifying CF-LIBS composition credible intervals. Unlike SBC's rank statistic, TARP needs ONLY posterior SAMPLES (no density evaluations), computes coverage from distances to random reference points, and is proven necessary-AND-sufficient for posterior accuracy. Works directly on the repo's NUTS/dynesty/grid-posterior draws and on the high-dimensional joint (T, n_e, C_1..C_K) where SBC's per-parameter rank histograms can miss joint miscalibration.
- **Expected benefit:** Detects inaccurate inferences in high-dimensional parameter spaces where SBC and other rank-based tests fail; yields a single expected-coverage-probability curve (nominal vs empirical) that is a publishable calibration certificate for the full composition vector, not just marginals. Pure numpy (random points + distance ranks).
- **Integration difficulty:** low
- **Citation:** *Sampling-Based Accuracy Testing of Posterior Estimators for General Inference*, Proceedings of the 40th International Conference on Machine Learning (ICML 2023); arXiv:2302.03026, 2023 — doi:10.48550/arXiv.2302.03026

### Conformal prediction over physics SURROGATES / emulators (manifold-aware coverage-guaranteed intervals) `[NEW]`
- **Applicability:** uncertainty (calibrated intervals) + forward model: extends split-CP/CQR (methods 4-5) from wrapping the point inversion to wrapping the PRE-COMPUTED MANIFOLD/EMULATOR itself. Calibrates the error the surrogate forward model introduces relative to the exact Saha-Boltzmann spectrum (and field-valued spectral outputs) with finite-sample coverage, so the repo's coarse-to-fine manifold path can ship coverage-guaranteed T/n_e/composition intervals rather than uncalibrated nearest-neighbour estimates.
- **Expected benefit:** Marginal coverage guarantee transferred to the emulator stage; the paper demonstrates calibrated intervals on physics field surrogates (plasma/fusion PDE emulators) with the conformal correction absorbing surrogate-vs-truth discrepancy -- directly the manifold-approximation error the CF-LIBS fast path currently ignores. Core CP algorithm is pure numpy/scipy.
- **Integration difficulty:** medium
- **Citation:** *Uncertainty quantification of surrogate models using conformal prediction*, Machine Learning: Science and Technology 5(4); arXiv:2408.09881, 2024 — doi:10.1088/2632-2153/ae2e7b

### Simulation-Based Inference with Neural Posterior Estimation (SBI-NPE) for spectral fitting with PCA reduction + calibrated amortized posteriors (knowledge / ML-only) `[NEW, phys✗]`
- **Applicability:** inversion-quantification + uncertainty + real-time (KNOWLEDGE-ONLY): the peer-reviewed, mature counterpart to the survey's 2021 LIBS normalizing-flow paper (method 8). Trains a flow on forward-model-simulated spectra folded through the instrument response to return amortized posteriors; introduces prior-range reduction via a classifier/coarse inference and shows PCA dimension reduction preserves posterior quality -- a blueprint for what a physics-only manifold/kernel-density posterior over the CF-LIBS manifold should reproduce. Demonstrated on real NICER instrument data with well-calibrated posteriors and <1 s/spectrum inference.
- **Expected benefit:** Amortized (sub-second) posteriors matching MCMC accuracy, robust to local-minima trapping, with PCA-reduced spectra giving no posterior degradation -- evidence that the repo's PCA + manifold path can carry calibrated UQ. Reported well-calibrated against frequentist/Bayesian X-ray fits.
- **Integration difficulty:** high
- **Citation:** *Simulation-based inference with neural posterior estimation applied to X-ray spectral fitting*, Astronomy & Astrophysics 686, A133; arXiv:2401.06061, 2024 — doi:10.1051/0004-6361/202449214

### Kennedy-O'Hagan Bayesian model calibration with GP model-discrepancy for ChemCam Mars LIBS composition `[NEW]`
- **Applicability:** inversion-quantification + uncertainty: the closest published analogue of a FULL Bayesian, discrepancy-aware composition retrieval on the exact target data (ChemCam Mars LIBS). Couples a physics-grounded forward model to a Kennedy-O'Hagan GP discrepancy term so the inferred composition posterior reflects model-form error, not just noise -- the same discrepancy idea as method 7 but at the composition (not just T/n_e) level. Directly informs the repo's SuperCam/ChemCam pds path and Bayesian solver design.
- **Expected benefit:** Calibrated composition posteriors that are not over-confident from an over-trusted Saha-Boltzmann/optically-thin model; demonstrates the discrepancy-term recipe on planetary LIBS specifically. GP discrepancy + MCMC implementable with numpy/scipy (no sklearn/torch needed for the GP kernel + likelihood).
- **Integration difficulty:** high
- **Citation:** *An initial exploration of Bayesian model calibration for estimating the composition of rocks and soils on Mars*, Statistical Analysis and Data Mining 13(6):524-542; arXiv:2008.04982, 2020 — doi:10.1002/sam.11503

## Real-time / differentiable / GPU spectral inversion: state-of-the-art methods applicable to the CF-LIBS forward model, inversion/quantification, uncertainty, and real-time GPU/JAX path

### Autodifferentiable line-by-line spectral forward model + HMC-NUTS (ExoJAX, Paper I)
- **Applicability:** forward model (differentiable Saha-Boltzmann -> spectrum) and inversion-quantification (gradient-based / HMC-NUTS recovery of T, n_e, composition). Maps to BayesianForwardModel + NumPyro NUTS and the joint L-BFGS-B optimizer.
- **Expected benefit:** Exact gradients for the forward model so NumPyro/L-BFGS gets correct cheap Jacobians instead of finite differences; HMC-NUTS scales to many-parameter joint fits with far fewer likelihood evals than nested sampling. Speed via the JAX jit/vmap/GPU path.
- **Integration difficulty:** medium
- **Citation:** *Autodifferentiable Spectrum Model for High-dispersion Characterization of Exoplanets and Brown Dwarfs*, Astrophysical Journal Supplement Series (ApJS 258, 31), 2022 — doi:10.3847/1538-4365/ac3b4d

### GPU memory-efficient differentiable opacity + multi-mode radiative transfer with native-resolution retrieval (ExoJAX2)
- **Applicability:** forward model (GPU-efficient differentiable opacity at full spectral resolution; atomic-line support is exactly the LIBS regime) and real-time (single-GPU native-resolution evaluation). Maps to the manifold/JAX forward-model path and radiation/SpectrumModel.
- **Expected benefit:** Removes the memory bottleneck that forces spectral binning, so high-resolution LIBS spectra can be forward-modeled and differentiated on one GPU without losing line information; atomic-database treatment is the directly relevant feature for LIBS (vs molecular-only Paper I).
- **Integration difficulty:** medium
- **Citation:** Differentiable Modeling of Planet and Substellar Atmosphere: High-resolution Emission, Transmission, and Reflection Spectroscopy with ExoJAX2, Astrophysical Journal (ApJ 985, 263), 2025 — doi:10.3847/1538-4357/adcba2

### Differentiable programming for plasma diagnostics: reverse-mode AD + GPU + batching for parameter estimation
- **Applicability:** inversion-quantification and real-time. Most direct plasma-physics precedent for turning a Saha-Boltzmann/emission forward model into a differentiable JAX diagnostic; the AD/GPU/batch speedup decomposition is a quantitative target for the CF-LIBS JAX inversion path.
- **Expected benefit:** Quantified ~140x end-to-end acceleration for plasma-parameter estimation (10x AD + 10x GPU + 1.4x batching). Establishes that gradient-based plasma diagnostics with O(10^3) parameters are feasible in seconds on a GPU.
- **Integration difficulty:** medium
- **Citation:** *Differentiable Programming for Plasma Physics: From Diagnostics to Discovery and Design*, arXiv preprint (arXiv:2603.11231), 2026 — S2:5ccfdaac0b79d75f47c78c072ced838b78b07de8

### Differentiable JAX X-ray spectral fitting with NUTS + variational inference + vmap batch fitting (jaxspec)
- **Applicability:** inversion-quantification, uncertainty, and real-time. Mirrors cflibs BayesianForwardModel + NumPyro NUTS + (potential) VI; demonstrates the differentiable-likelihood + gradient-sampler + GPU-jit stack the project targets.
- **Expected benefit:** ~10x faster than existing non-differentiable spectral-fitting alternatives; variational inference produces usable posteriors even on high-resolution data in under ~10 minutes on a GPU. Provides a vetted design for adding VI as a fast UQ option alongside NUTS.
- **Integration difficulty:** medium
- **Citation:** *jaxspec: A fast and robust Python library for X-ray spectral fitting*, Astronomy & Astrophysics (A&A 690, A317), 2024 — doi:10.1051/0004-6361/202451736

### Amortized inference via Neural Posterior Estimation (normalizing flows) for spectral retrieval `[phys✗]`
- **Applicability:** inversion-quantification + uncertainty + real-time (amortized batch/streaming inference). A fast front-end returning a full posterior over (T, n_e, composition) per spectrum; complements the physics solver as an initializer or real-time first pass.
- **Expected benefit:** Per-spectrum inference reduced from hours (nested sampling) to a few seconds with coverage-tested posteriors; amortizes over thousands of spectra (batch/streaming LIBS).
- **Integration difficulty:** high
- **Citation:** *Neural posterior estimation for exoplanetary atmospheric retrieval*, Astronomy & Astrophysics (A&A 672, A147), 2023 — doi:10.1051/0004-6361/202245263

### Gaussian-process surrogate/emulator of the radiative-transfer forward model `[DOI-corrected]`
- **Applicability:** forward model (emulator/surrogate) + inversion (cheap Jacobians for the solver) + real-time. A continuous, differentiable surrogate with built-in approximation-uncertainty; alternative to the FAISS-on-manifold path.
- **Expected benefit:** Forward-model evaluation accelerated by orders of magnitude while staying within ~1% relative error; emulator Jacobians enable fast gradient-based inversion.
- **Integration difficulty:** medium
- **Citation:** *Forward model emulator for atmospheric radiative transfer using Gaussian processes and cross validation*, Atmospheric Measurement Techniques (AMT 18, 673), 2025 — doi:10.5194/amt-18-673-2025

### Generalized Humlicek rational approximation for the Voigt/complex error function (vectorizable)
- **Applicability:** forward model (line broadening: Voigt = Gaussian Doppler conv. Lorentzian Stark). Replaces/augments the current Gaussian/Doppler profiles in cflibs/radiation for accurate Stark-broadened Voigt lineshapes in a vectorized jax.numpy path.
- **Expected benefit:** Accurate Voigt (1e-5 to 1e-6 rel. error) at near-rational-function cost; vectorizes cleanly across wavelength grid x lines for GPU batch evaluation; no slow Faddeeva call per point.
- **Integration difficulty:** low
- **Citation:** *The Voigt and complex error function: Humlicek's rational approximation generalized*, Monthly Notices of the Royal Astronomical Society (MNRAS 479, 3068), 2018 — doi:10.1093/mnras/sty1680

### Algorithm 916: arbitrary-accuracy, tunable Faddeyeva/Voigt evaluation
- **Applicability:** forward model (high-accuracy Voigt reference + tunable speed/accuracy) and uncertainty/validation (ground-truth lineshape for parity tests against the fast approximation).
- **Expected benefit:** Controllable accuracy with explicit error bound; >2x runtime improvement at top accuracy via the Remark. Validation oracle for the fast Voigt path; fallback when 1e-6+ accuracy is required.
- **Integration difficulty:** low
- **Citation:** *Algorithm 916: Computing the Faddeyeva and Voigt Functions*, ACM Transactions on Mathematical Software (TOMS 38(2), 15), 2011 — doi:10.1145/2049673.2049679

### Analytic Voigt-Hjerting approximation (Tepper-Garcia 2006) for autodiff-friendly lineshapes
- **Applicability:** forward model (differentiable Voigt lineshape for the JAX/manifold path) and inversion (clean gradients through the lineshape for HMC/L-BFGS). Drop-in for cflibs/radiation profile broadening in the differentiable branch.
- **Expected benefit:** ~1e-4 accuracy for a<=1 at the cost of one Gaussian + a low-order polynomial; fully differentiable and vectorizable, removing the special-function-call bottleneck and giving exact analytic gradients for gradient-based inversion.
- **Integration difficulty:** low
- **Citation:** *Voigt Profile Fitting to Quasar Absorption Lines: An Analytic Approximation to the Voigt-Hjerting Function*, Monthly Notices of the Royal Astronomical Society (MNRAS 369, 2025), 2006 — doi:10.1111/j.1365-2966.2006.10450.x

### CSigma graphs (Cigma-LIBS) for joint multi-line, multi-element calibration-free plasma characterization with self-absorption handling `[DOI-corrected]`
- **Applicability:** inversion-quantification and self-absorption. Augments/replaces the per-element Boltzmann-plot + closure with a joint multi-element cross-section fit; the inhomogeneity/columnar-density variant maps onto the project's CDSB self-absorption module.
- **Expected benefit:** More robust joint estimation of T, n_e and concentrations from all lines simultaneously, with self-absorption encoded in the cross-section; the columnar-density/inhomogeneity variant corrects inhomogeneity bias that degrades standard CF-LIBS accuracy.
- **Integration difficulty:** medium
- **Citation:** *CSigma graphs: A new approach for plasma characterization in laser-induced breakdown spectroscopy*, Journal of Quantitative Spectroscopy and Radiative Transfer (JQSRT 149, 90-102), 2014 — doi:10.1016/j.jqsrt.2014.07.026

### JAX-COSMO: end-to-end differentiable and GPU-accelerated physics inference library (architectural precedent) `[NEW]`
- **Applicability:** real-time + inversion-quantification + uncertainty: a vetted reference design for the whole CF-LIBS JAX path. Shows how to make a complete physics forward model end-to-end differentiable and GPU-accelerated so HMC/Fisher/gradient inference run on the same jit-compiled graph as the manifold generator. Directly informs cflibs/manifold + BayesianForwardModel architecture.
- **Expected benefit:** Demonstrates Fisher-matrix uncertainty and HMC inference at orders-of-magnitude speedup over finite-difference pipelines by reusing one differentiable forward model for both posterior sampling and forecasting; the closest end-to-end blueprint for the project's 'one JAX forward model drives manifold + Bayesian + L-BFGS' goal.
- **Integration difficulty:** medium
- **Citation:** *JAX-COSMO: An End-to-End Differentiable and GPU Accelerated Cosmology Library*, The Open Journal of Astrophysics, 2023 — doi:10.21105/astro.2302.05163

### Monte-Carlo full-spectrum calibration-free LIBS with GPU acceleration (Gornushkin & Voelker) + closed-form Planck self-absorption correction `[NEW]`
- **Applicability:** inversion-quantification (primary solver) + forward model + real-time GPU path: a full-spectrum radiative-transfer CF solver fit by global stochastic optimization, with the line-based Boltzmann pipeline as initializer. The GPU ~5 min/spectrum cost and embarrassingly-parallel Monte-Carlo map directly onto the repo's JAX vmap/manifold path. The closed-form companion SA correction I_thin = -B_lambda ln(1 - I/B_lambda) is a cheap, vectorizable, differentiable drop-in for the radiation module.
- **Expected benefit:** Synthetic 8-element slag: ~1% intrinsic relative accuracy (~10x better than Boltzmann-plot CF) uniform across concentration ranges, on GPU; closed-form SA correction reaches T to 0.4% and n_e to 0.7% RSD in 4 iterations for tau0<=2-3. The Monte-Carlo evaluation is exactly the batched-forward-model workload the JAX GPU path is built for.
- **Integration difficulty:** high
- **Citation:** Intrinsic Performance of Monte Carlo Calibration-Free Algorithm for Laser-Induced Breakdown Spectroscopy (companion: Investigation of a Method for the Correction of Self-Absorption by Planck Function in LIBS), Sensors 22(19):7149 (companion: Journal of Analytical Atomic Spectrometry, 2023, DOI 10.1039/d2ja00352j, paperId 0a05b647095255ae08edec358b74f6af16b0de7b), 2022 — doi:10.3390/s22197149

### microJAX: fully differentiable, JIT/GPU-accelerated physics forward model with NumPyro samplers (recent differentiable-modeling design pattern) `[NEW]`
- **Applicability:** forward model + inversion-quantification + real-time: a 2025 design precedent (forward-cites both ExoJAX2 and jaxspec) showing a domain physics forward model made fully differentiable on JAX/XLA with exact AD gradients and GPU JIT, then sampled with gradient-based inference. Reinforces the recommended pattern for cflibs SpectrumModel -> exact-gradient inversion.
- **Expected benefit:** Exact gradients through a previously non-differentiable physics kernel via JAX + XLA JIT, with GPU parallelism; demonstrates that even geometrically complex forward models become amenable to HMC/L-BFGS once expressed in JAX -- evidence the CF-LIBS forward model can follow the same route for cheap, correct Jacobians.
- **Integration difficulty:** medium
- **Citation:** *microJAX: A Differentiable Framework for Microlensing Modeling with GPU-accelerated Image-centered Ray Shooting*, Astrophysical Journal (arXiv:2510.02639), 2025 — doi:10.3847/1538-4357/ae1005

### Neural Posterior Estimation (SBI) for X-ray spectral fitting -- amortized real-time inference precedent in the jaxspec ecosystem `[NEW, phys✗]`
- **Applicability:** inversion-quantification + uncertainty + real-time: a newer, spectral-fitting-specific (rather than exoplanet-specific) amortized-inference precedent that pairs with jaxspec's differentiable simulator to train an NPE/flow front-end. Maps to a fast, batched first-pass posterior over (T, n_e, composition) for streaming LIBS, with the physics solver as refiner/validator.
- **Expected benefit:** Amortized posteriors in seconds per spectrum after a one-time training on simulator draws; coverage-testable. Strengthens method 4 (exoplanet NPE) with a closer-domain (X-ray spectral fitting) demonstration and the jaxspec differentiable-simulator-as-training-engine pattern.
- **Integration difficulty:** high
- **Citation:** *Simulation-based inference with neural posterior estimation applied to X-ray spectral fitting*, Astronomy & Astrophysics (follow-up: A&A 2025, DOI 10.1051/0004-6361/202555215, paperId not separately fetched), 2024 — doi:10.1051/0004-6361/202449214

### Real-time calibration-free LIBS supported by machine learning (closest direct LIBS real-time comparator) `[NEW, phys✗]`
- **Applicability:** inversion-quantification + real-time: the most on-topic recent LIBS-specific paper for the 'real-time CF-LIBS' theme. Learns a map from spectra to plasma state / composition for near-instant inference tolerant of non-ideal LTE. Serves as the speed/robustness comparator the physics-only JAX path is benchmarked against.
- **Expected benefit:** Near-instant T/n_e/composition on large spectrum sets; reported accuracy comparable to classical CF-LIBS when training coverage is good. Quantifies the real-time speed ceiling a learned surrogate reaches.
- **Integration difficulty:** high
- **Citation:** *Towards real-time calibration-free LIBS supported by machine learning*, Spectrochimica Acta Part B: Atomic Spectroscopy, 2024 — doi:10.1016/j.sab.2024.107082

## Atomic-data quality and mathematical robustness of the CF-LIBS inversion: best transition-probability/partition-function databases, atomic-data error budgets, and ill-posed-inverse-problem math (Tikhonov regularization, NNLS, errors-in-variables Boltzmann fitting, compositional-data/closure handling)

### Barklem & Collet (2016) partition functions and dissociation equilibrium constants for atoms (H-U) and 291 diatomics
- **Applicability:** Forward model + inversion-quantification: supplies U_s(T) for the Saha-Boltzmann population link and closure normalization; bounds systematic error in absolute number densities and ion-stage ratios. Already integrated via patch_partition_functions_bc2016.py.
- **Expected benefit:** Removes an uncontrolled systematic; truncated-sum vs full-set U(T) differences reach several to tens of percent for open-shell transition metals (Fe, Ti) at LIBS temperatures (0.7-1.2 eV), mapping ~1:1 onto concentration bias. Make canonical default and validate high-T tail.
- **Integration difficulty:** low
- **Citation:** *Partition functions and equilibrium constants for diatomic molecules and atoms of astrophysical interest*, Astronomy & Astrophysics (588, A96; arXiv:1602.03304), 2016 — doi:10.1051/0004-6361/201526961

### NIST Atomic Spectra Database (ASD) with accuracy-graded transition probabilities + grade-aware inverse-variance weighting `[DOI-corrected]`
- **Applicability:** Identification + inversion-quantification: A_ki/g/E_k inputs to Boltzmann/Saha fits; the NIST accuracy grade gives a principled per-line weight and a hard quality filter. Repo already has aki_uncertainty_weighting in boltzmann.py.
- **Expected benefit:** Restricting Boltzmann fits to A/B-graded lines (<=10%) and inverse-variance weighting the rest tightens fitted T and reduces concentration scatter. SOTA step: drive weights from the actual NIST grade-to-sigma table, expose grade as a line-selection gate.
- **Integration difficulty:** low
- **Citation:** NIST Atomic Spectra Database (ver. 5.x) [dataset, DOI 10.18434/T4W30F] + 'Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium' (grade-to-sigma definitions), NIST (Gaithersburg) for the ASD dataset; J. Phys. Chem. Ref. Data 38:565 (2009) for grade definitions, 2009 — doi:10.18434/T4W30F (dataset; not S2-indexed) | grade defs: 10.1063/1.3077727

### Monte-Carlo atomic-data error budget (random A_ki/E_k trials propagated through the full inversion)
- **Applicability:** Uncertainty quantification: turns NIST accuracy grades into a full output error budget on T, n_e, and concentrations; flags atomic-data-limited lines for line selection. Repo has MonteCarloUQ in uncertainty.py.
- **Expected benefit:** Honest atomic-data-aware error bars (the largest CF-LIBS systematic) plus a sensitivity ranking to prune worst lines. Addition: perturb A_ki/E_k from grade-derived sigmas, not just intensities/noise.
- **Integration difficulty:** medium
- **Citation:** *Assessing Uncertainties of Theoretical Atomic Transition Probabilities with Monte Carlo Random Trials*, Atoms (2(2):86-122), 2014 — doi:10.3390/atoms2020086

### Errors-in-variables / weighted total-least-squares (orthogonal distance) Boltzmann-plot regression `[DOI-corrected]`
- **Applicability:** Inversion-quantification: replace the OLS Boltzmann/Saha-Boltzmann slope fit with weighted-TLS/ODR for an unbiased T and correct slope covariance feeding uncertainty propagation. Drop-in via scipy.odr.
- **Expected benefit:** Removes OLS slope bias (worst when E_k range is narrow or A_ki spread large), yields statistically consistent T with proper covariance, improving trueness of T and every concentration. Adds sigma_x to the existing inverse-variance WLS + robust variants in boltzmann.py.
- **Integration difficulty:** medium
- **Citation:** Plasma excitation temperature obtained with Boltzmann plot method: Significance, precision, trueness and accuracy (Safi et al.); + Van Huffel & Vandewalle TLS (1991); + Boggs/Byrd/Schnabel ODRPACK (1987), Spectrochimica Acta Part B: Atomic Spectroscopy (206, 2023), 2023 — doi:Safi LIBS Boltzmann-plot accuracy paper: 10.1016/j.sab.2023.106686 (survey said .106723 - WRONG) | TLS: 10.1137/1.9781611971002 | ODRPACK: 10.1137/0908085

### CSigma (C-sigma) graphs: self-absorption-aware multi-element, multi-line generalized curve-of-growth inversion `[DOI-corrected]`
- **Applicability:** Self-absorption + inversion-quantification: joint multi-element/multi-line fit with optical depth built into the model, replacing separate Boltzmann + curve-of-growth self-absorption stages. Repo has a self_absorption.py / CSigma-graph path.
- **Expected benefit:** Improved analytical accuracy for major/minor elements vs standard CF-LIBS on alloys and fused-glass rock standards by removing self-absorption bias and stabilizing the common slope.
- **Integration difficulty:** medium
- **Citation:** *CSigma graphs: A new approach for plasma characterization in laser-induced breakdown spectroscopy*, Journal of Quantitative Spectroscopy and Radiative Transfer (149:90-102), 2014 — doi:10.1016/j.jqsrt.2014.07.026 | comparison sub-cite: 10.1021/acs.analchem.9b01885

### Tikhonov-regularized / NNLS spectral intensity unmixing for blended-line line-intensity extraction
- **Applicability:** Preprocess + identification: regularized/NNLS deconvolution of overlapping lines into clean per-transition intensities feeding the Boltzmann fit; L-curve/GCV pick lambda without ML. scipy.optimize.nnls / scipy.linalg only.
- **Expected benefit:** Stabilizes line-intensity extraction in crowded spectra (transition metals, oxide bands), reducing interference-induced Boltzmann outliers; NNLS gives sparse physically-valid (x>=0) intensities.
- **Integration difficulty:** medium
- **Citation:** Analysis of Discrete Ill-Posed Problems by Means of the L-Curve (Hansen); Solving Least Squares Problems (Lawson & Hanson, NNLS); Generalized cross-validation... (Golub, Heath, Wahba); Tikhonov & Arsenin (1977), SIAM Review 34(4):561-580 (Hansen); Technometrics 21(2):215-223 (GCV); SIAM Classics (Lawson-Hanson), 1992 — doi:L-curve: 10.1137/1034115 | NNLS book: 10.1137/1.9781611971217 (SIAM reprint) | GCV: 10.1080/00401706.1979.10489751

### Compositional data analysis: Aitchison geometry with isometric log-ratio (ILR) / pivot (PWLR) coordinates for the Sum C = 1 closure
- **Applicability:** Inversion-quantification (closure) + uncertainty: enforce Sum C=1 by unconstrained optimization in ILR/PWLR space; report covariance in log-ratio coordinates; handle below-LOD elements via CDA zero-replacement. Repo has ILR + PWLR + softmax closure modes in closure.py.
- **Expected benefit:** Eliminates closure-induced negative-correlation artifacts and out-of-simplex iterates; full-rank covariance vs rank-deficient covariance of raw closed concentrations; meaningful (Aitchison-distance) error bars. SOTA additions: propagate uncertainty IN ILR coordinates; principled below-LOD zero handling per the 2023 reappraisal.
- **Integration difficulty:** low
- **Citation:** Isometric Logratio Transformations for Compositional Data Analysis (Egozcue et al.); Aitchison's Compositional Data Analysis 40 Years On: A Reappraisal (Greenacre et al.); Linear regression with compositional explanatory variables (Hron et al.); The Statistical Analysis of Compositional Data (Aitchison), Mathematical Geology 35(3):279-300 (Egozcue 2003); Statistical Science 38(3):386-410 (Greenacre, print 2023; S2 indexes arXiv preprint as 2022), 2003 — doi:ILR: 10.1023/A:1023818214614 | reappraisal: 10.1214/22-STS880 | PWLR: 10.1080/02664763.2011.644268 | Aitchison 1982: 10.1111/j.2517-6161.1982.tb01195.x

### Tognoni et al. CF-LIBS state-of-the-art critical review as the methodological reference frame
- **Applicability:** Cross-cutting (identification + inversion + self-absorption + UQ): defines line-selection and LTE/optical-thinness gates and the error-source taxonomy the other methods operationalize. Drives line_selection.py and lte_validator.py defaults.
- **Expected benefit:** Not a numeric gain but the validated rule-set (energy-span, interference, accuracy-grade filters; LTE/McWhirter checks) whose enforcement is the difference between CF-LIBS results within a few percent vs tens of percent of nominal.
- **Integration difficulty:** low
- **Citation:** *Calibration-Free Laser-Induced Breakdown Spectroscopy: State of the art*, Spectrochimica Acta Part B 65(1):1-14, 2010 — doi:10.1016/j.sab.2009.11.006 | foundational method: 10.1366/0003702991947612

### Sparse / shrinkage regression (LASSO, elastic net) for matrix-effect-robust quantification (KNOWLEDGE only; not shippable as-is) `[phys✗, DOI-corrected]`
- **Applicability:** Identification / line-selection (as knowledge): motivates sparsity priors; the L1 idea informs the physics-only NNLS/Tikhonov unmixing, but sklearn-based estimators are barred from the shipped path.
- **Expected benefit:** Improved predictive accuracy and automatic line selection under matrix effects in calibration-based settings; transferable idea for the physics-only pipeline is the sparsity prior (realized via NNLS / L1-regularized scipy solves), not the sklearn estimators.
- **Integration difficulty:** high
- **Citation:** Comparison of partial least squares and lasso regression techniques as applied to laser-induced breakdown spectroscopy of geological samples (Anderson et al.); Tibshirani LASSO (1996); Zou & Hastie elastic net (2005), Spectrochimica Acta Part B 70:24-32, 2012 — doi:Anderson LASSO-LIBS: 10.1016/j.sab.2012.04.011 (survey said .04.004 - WRONG) | elastic net: 10.1111/j.1467-9868.2005.00503.x | LASSO: 10.1111/j.2517-6161.1996.tb02080.x

### Aguilera & Aragon (2024) inhomogeneity-aware CSigma procedure `[NEW]`
- **Applicability:** Self-absorption + inversion-quantification (forward model): extends the C-sigma graph (verified Method 4) to relax the single-zone-uniform-plasma assumption by modeling laser-induced plasma inhomogeneity, the next dominant systematic after self-absorption in the C-sigma / generalized-curve-of-growth pipeline. Maps onto the repo's self_absorption.py / CSigma-graph path.
- **Expected benefit:** Reduces the inhomogeneity bias that the standard single-zone CSigma (and standard Saha-Boltzmann CF-LIBS) cannot capture; directly improves trueness on strongly self-absorbed, spatially inhomogeneous plasmas (resonance-line-dominated rock/alloy matrices). Authored by the original CSigma authors, so it is the canonical successor.
- **Integration difficulty:** high
- **Citation:** *New procedure for CSigma laser induced breakdown spectroscopy addressing the laser-induced plasma inhomogeneity*, Spectrochimica Acta Part B: Atomic Spectroscopy, 2024 — doi:10.1016/j.sab.2024.106969

### Zhang et al. (2024) partition-function-cutoff vs ionization-potential-lowering consistency study `[NEW]`
- **Applicability:** Forward model (Saha-Boltzmann normalization) + uncertainty: quantifies how the high-Rydberg partition-sum cutoff and the Debye/Stewart-Pyatt ionization-potential lowering jointly (and inconsistently, if mismatched) bias the spectroscopic temperature and ion-stage balance in an aluminum (LIBS-relevant) plasma. Directly operationalizes the 'use ONE consistent Delta-chi for U(T) and the Saha continuum lowering' requirement underlying verified Method 0 and the IPD consistency theme. Feeds partition.py + saha_boltzmann.py.
- **Expected benefit:** Bounds and removes the otherwise-ignored systematic from a mismatch between the partition cutoff and the Saha ionization-energy reduction; gives the quantitative T-bias as a function of cutoff choice, enabling a defensible single-cutoff convention rather than two independent ad-hoc choices.
- **Integration difficulty:** low
- **Citation:** *Influences of Partition Function Cutoff Versus Lowering of Ionization Energy on Spectroscopic Temperature Measurement in Aluminum Plasmas*, IEEE Transactions on Plasma Science, 2024 — doi:10.1109/TPS.2024.3452482

### Argon-plasma partition-function-cutoff sensitivity for spectral temperature (2024) `[NEW]`
- **Applicability:** Forward model (Saha-Boltzmann normalization) + uncertainty: complementary to the IEEE/aluminum study; isolates the sensitivity of the measured temperature to the partition-sum truncation choice. Supports validating the high-T tail of the Barklem-Collet U(T) used in partition.py (verified Method 0's explicit recommendation).
- **Expected benefit:** Provides a quantitative test of how much partition-cutoff choice perturbs the fitted T (hence every concentration via the Boltzmann/Saha chain), giving an empirical basis for the partition-tail validation the survey recommends but did not quantify.
- **Integration difficulty:** low
- **Citation:** *The effects of partition function cutoff on spectral temperature measurement in argon plasma*, AIP Advances, 2024 — doi:10.1063/5.0202284

### SuperCam-on-Mars LIBS plasma-diagnostics / quantification study (2024) `[NEW]`
- **Applicability:** Cross-cutting (inversion-quantification + LTE diagnostics + uncertainty), directly on the repo's target instrument: characterizes the actual plasma T/n_e regime, LTE validity, and the atomic-data/self-absorption error budget for SuperCam on Mars, defining which of the verified robustness measures (grade-aware weighting, EIV Boltzmann fit, self-absorption-aware C-sigma, consistent partition/IPD cutoff) actually matter for ChemCam/SuperCam-class spectra.
- **Expected benefit:** Grounds the whole topic's robustness toolkit in the real ground-truth plasma conditions of the planetary use case (cflibs/pds/), telling the pipeline which lines/elements are atomic-data- or LTE-limited under Martian-atmosphere LIBS, and provides a 2024 reference dataset/regime for validating the inversion's uncertainty budget.
- **Integration difficulty:** medium
- **Citation:** *LIBS plasma diagnostics with SuperCam on Mars: Implications for quantification of elemental abundances*, Spectrochimica Acta Part B: Atomic Spectroscopy, 2024 — doi:10.1016/j.sab.2024.107061
