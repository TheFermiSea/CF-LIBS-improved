# Audit 02 — Inversion / Solver Physics (CF-LIBS core)

**Date:** 2026-06-09 · **Scope:** `cflibs/inversion/physics/`, `cflibs/inversion/solve/`, supporting
`cflibs/radiation/stark.py`, `cflibs/inversion/identify/line_detection.py` (intensity extraction only).
**Method:** read-only code audit + small numeric checks (`JAX_PLATFORMS=cpu`) against the production DB
(`ASD_da/libs_production.db`), grounded in peer-reviewed CF-LIBS literature.
**Instrument context:** 1 ps Yb:fiber @1040 nm; ps-LIBS regime T ≈ 0.5–1.3 eV, n_e ≈ 1e16–1e18 cm⁻³.

Findings are **ranked by expected composition-accuracy impact** on real rock/basalt data.

---

## Executive summary

The Ciucci-1999 skeleton is implemented correctly: `y = ln(Iλ/gA)` is the right ordinate
(`cflibs/inversion/common/data_structures.py:54-63`), the multi-species Saha shift to the neutral plane has
the correct form (`cflibs/inversion/solve/iterative.py:1252-1317`), the closure is
`C_s ∝ U_s(T)·exp(q_s)` (`cflibs/inversion/physics/closure.py:553-583`), and — answering the top suspect
directly — **the element total *does* include the ion stage**: every closure call is fed per-element
multipliers `M_s = 1 + n_II/n_I` from the Saha ratio (`iterative.py:1123-1158`, applied at
`closure.py:565,630,699`). The Fe/Al over-attribution is **not** a missing-ion-fraction bug.

The dominant accuracy losses are instead: **(F1)** a wrong hard-coded U_II fallback (Na II = 15 instead of
≈1.0 — a ~15× direct multiplier error on a basalt major); **(F2)** electron density that is never measured
(1-atm pressure balance instead of Stark widths) feeding the Saha multipliers; **(F3)** temperature bias
amplified differentially through U_s(T) (U_Fe doubles per +0.4 eV while U_Si rises 20%); **(F4)** a
self-absorption "correction" that is a positive-feedback loop on the recovered composition; **(F5)** no
spectral-response calibration hook anywhere in the inversion intensity path; and **(F6)** an
intensity-extraction floor that silently swaps integrated areas for peak heights. The SB-graph "fix" is
shown (numerically, to 1e-14) to be **algebraically identical** to the default common-slope plane under
equal weights — its measured benefit comes entirely from abandoning the w∝I weighting, which means the
`boltzmann_weight_cap=5` band-aid and the SB-graph are two partial treatments of one root cause (F7).

---

## F1 — Hard-coded U_II fallback = 15.0 is wrong by ~15× for Na II (and Li II, H II)

**Impact: critical (direct, multiplicative, on a basalt major).**

**(a) Code behavior.** `_compute_abundance_multipliers` uses
`U_II = partition_funcs_II.get(el, 15.0)` (`iterative.py:1149`), and the partition ladder bottoms out at
`_canonical_partition_fallback(stage=2) → 15.0`
(`cflibs/inversion/physics/self_absorption_inputs.py:72-78`). Audit check against
`ASD_da/libs_production.db`: stage-II partition providers are **missing for H, Li, Na** — every other
element resolves. So at T = 1 eV the solver uses U(Na II) = 15.0. The true value is ≈ 1.00 (Ne-like closed
shell; first excited level ≈ 33 eV). Verified in-repo: K II (Ar-like, physically the same situation) *does*
have levels and correctly evaluates to U = 1.0000.

**(b) Literature requirement.** The Saha ratio `n_II/n_I = (2/n_e)(2πm_ekT/h²)^{3/2}(U_II/U_I)exp(−χ/kT)`
requires the actual partition functions (Ciucci et al. 1999, *Appl. Spectrosc.* 53, 960,
doi:10.1366/0003702991947612; Tognoni et al. 2010, *Spectrochim. Acta B* 65, 1,
doi:10.1016/j.sab.2009.11.006, list partition-function accuracy in the CF-LIBS error budget).

**(c) Quantified impact.** At T = 1 eV, n_e = 1e17 cm⁻³, Na (χ=5.14 eV, U_I=5.33 from DB) is ≳99% ionized:
S_Na ∝ U_II. Using 15.0 instead of ~1.0 inflates `M_Na = 1+S_Na` — and hence C_Na before normalization —
by **~15×**. This alone can explain the historical "Na = 98 wt%" blow-ups, and it still silently taxes
every solve that includes Na (renormalization then suppresses all other elements).

**(d) Fix.** (1) Ingest Na II / Li II energy levels (NIST ASD has them) or store polynomial U(T) rows;
(2) replace the constant stage-II fallback with a physics-aware default (closed-shell ions → g₀ ≈ 1), or at
minimum emit a loud per-element warning when a closure-relevant species hits the fallback;
(3) add a regression test asserting U(Na II, 1 eV) < 2.

---

## F2 — Electron density is never measured: 1-atm pressure balance drives the Saha multipliers

**Impact: critical for ps-LIBS (T and n_e are the two plasma inputs; one is currently invented).**

**(a) Code behavior.** The canonical diagnostic exists (`StarkDiagnosticLine`, `iterative.py:127-160`;
`estimate_ne_from_stark`, `cflibs/radiation/stark.py:215-273`, with correct Olivero–Longbothum Voigt
deconvolution) but **no CLI/analysis path ever constructs one** — `invert`/`analyze` call
`solver.solve(observations, closure_mode=...)` with no `stark_diagnostic` (`cflibs/cli/main.py:540,623-630`).
Every production solve therefore takes the fallback: an isobaric **1-atm STP pressure balance**
(`_pressure_balance_ne`, `iterative.py:1749-1777`), which the code itself labels "physically invalid for
LIBS". The resulting n_e (≈0.6–3e17 cm⁻³ at 1 eV depending on Z̄) is uncorrelated with the actual plasma.
A supplied Stark line also forces the Python path — the lax path has no Stark support
(`iterative.py:1632-1641`).

**(b) Literature requirement.** Stark broadening of a well-characterized line (Hα, or Fe I/Si II lines) is
the standard n_e diagnostic feeding the Saha terms in CF-LIBS (Ciucci et al. 1999; Tognoni et al. 2010 §3;
Aragón & Aguilera, *Spectrochim. Acta B* 63 (2008) 893, doi:10.1016/j.sab.2008.05.010; Cristoforetti et
al. 2010, *Spectrochim. Acta B* 65, 86, doi:10.1016/j.sab.2009.11.005, require n_e to verify the
McWhirter/LTE criterion the whole algorithm assumes).

**(c) Quantified impact.** S ∝ 1/n_e. Where all majors are strongly ionized (S≫1) the error is largely
common-mode and cancels in the closure; the damage is *differential* in the partially-ionized regime,
exactly where a cooler ps-LIBS plasma sits. At T = 0.7 eV (DB partition functions: S_Fe(1e17) ≈ 0.4,
S_Al ≈ 1.2 — both in the steep `1+S` zone), moving n_e by ×10 moves M_Al/M_Si-type ratios by factors of
~2–5. Al's low χ (5.99 eV) and tiny U_II (1.08) make C_Al hypersensitive to (T, n_e) — a credible
contributor to the Al 19-vs-7.1 residual.

**(d) Fix.** Wire `stark_diagnostic` into `analyze`/`invert` (config: line λ, stored `stark_w` reference
from the `lines` table, instrument FWHM); prefer Hα when the range covers 656 nm. Treat `ne_from_stark`
(already in quality metrics, `iterative.py:2052-2055`) as a hard quality gate: a pressure-balance-n_e
solve should not be labeled quantitative. Cross-check the LTE validator's McWhirter output against the
*measured* n_e.

---

## F3 — Temperature bias multiplies back through U_s(T): the real "partition functions multiply back" mechanism

**Impact: high — this, not a closure-form error, is the Fe (and Ti) failure channel.**

**(a) Code behavior.** Closure computes `C_s ∝ M_s·U_s(T)·exp(q_s)` (`closure.py:565`). The fitted
intercept is `q_s = ln(n_I·F̃/U_I(T_true))` — the *true* U is embedded in the data; the closure multiplies
by `U_I(T_solver)`. The U factors cancel only if `T_solver = T_true`. (Also a one-iteration lag:
`partition_funcs` are evaluated at the pre-update T (`iterative.py:1846`) while intercepts come from the
current fit — harmless at convergence, inconsistent on non-converged exits.)

**(b) Literature requirement.** Ciucci et al. 1999 define `q_s = ln(C_sF/U_s(T))`; Tognoni et al. 2010 §4
flag temperature error → U(T) and exp-factor error as a leading CF-LIBS error source. The multi-element
common-slope fit is the correct mitigation (Aguilera & Aragón 2007, *Spectrochim. Acta B* 62, 378,
doi:10.1016/j.sab.2007.03.024).

**(c) Quantified impact** (direct-sum U from the production DB):

| element | U_I(0.8 eV) | U_I(1.0 eV) | U_I(1.2 eV) | ratio 1.2/0.8 |
|---|---|---|---|---|
| Fe | 52.3 | 75.5 | 106.5 | **2.04** |
| Ti | 66.9 | 97.1 | 132.0 | **1.97** |
| Si | 10.9 | 11.8 | 13.1 | 1.20 |
| Al | 6.2 | 7.0 | 8.3 | 1.33 |

A +0.4 eV T bias inflates C_Fe/C_Si by ~70% and C_Ti/C_Si by ~65% *through the partition functions
alone*, before the slope–intercept covariance term `δq_s = −x̄_s·δm` (with SB-graph ion lines Fe's x̄ is
10–15 eV, so this term is large too). Both push the same direction as the observed Fe ≈ 3–4× over.
Self-absorption of retained resonance lines (see F4; the CLI keeps resonance lines by default,
`cli/main.py:319-328`) *flattens* the slope → T biased high → exactly this amplification.

**(d) Fix.** (1) Treat T accuracy as the primary target: use the SB-graph (ion lines, long lever arm) for
the slope and **validate** with the Saha–Boltzmann consistency and per-element T spread checks
(thresholds at `cflibs/inversion/physics/quality.py:77-83` match accepted practice — wire them into
`converged`). (2) Propagate σ_T into U_s(T) in the uncertainty path (currently exact —
`uncertainty.py:232-282`). (3) Consider one-point calibration to absorb residual T/response systematics
(Cavalcanti et al. 2013, *Spectrochim. Acta B* 87, 51, doi:10.1016/j.sab.2013.05.016).

---

## F4 — Self-absorption correction is a positive-feedback loop on the recovered composition

**Impact: high when enabled (explains "SA correction makes intercept inflation WORSE").**

**(a) Code behavior.** The implemented correction is a Bulajic-flavored escape-factor divide: τ₀ from the
classical Doppler line-center formula (`cflibs/inversion/physics/self_absorption.py:1040-1219` —
dimensionally correct, prefactor verified in the docstring), `I_thin = I_obs/f(τ)`,
`f(τ)=(1−e^{−τ})/τ`, τ capped at 10, re-run every solver iteration (`iterative.py:1850-1872`). Crucially
**τ is computed from the *recovered* concentrations** (`self_absorption.py:1120,1137`:
`n_s = C_s·n_tot`), with no observable gate. Neither internal-reference (IRSAC) nor doublet/duplication
methods drive the loop — the doublet-ratio machinery exists (`correct_via_doublet_ratio`,
`find_doublet_pairs`, `self_absorption.py:332-540`) but is **not wired into the solver**. `cdsb.py` is
data-structures only (the CD-SB plotter was removed).

**(b) Literature requirement.** Bulajic et al. 2002 (*Spectrochim. Acta B* 57, 339,
doi:10.1016/S0584-8547(01)00398-6) iterate τ against a curve-of-growth model constrained by *measured*
line profiles/widths, not the composition estimate alone; Sun & Yu 2009 (*Talanta* 79, 388,
doi:10.1016/j.talanta.2009.03.066) anchor per-line SA factors to an observed high-E_k internal-reference
line; El Sherbini et al. 2005 (*Spectrochim. Acta B* 60, 1573, doi:10.1016/j.sab.2005.10.011) derive the
SA coefficient from the measured/Stark width ratio; Cristoforetti & Tognoni 2013 (*Spectrochim. Acta B*
79–80, 63, doi:10.1016/j.sab.2012.11.010) extract columnar density *from the self-absorbed line shapes
themselves*. Common denominator: the correction must be anchored to an **observable**, never only to the
current composition iterate.

**(c) Why it inflates intercepts (diagnosis).** Over-attributed element (Fe) → larger C_Fe → larger τ for
*every* Fe line → larger 1/f(τ) boost → q_Fe rises → closure gives Fe even more mass → next iteration
boosts again. The τ-cap merely saturates the runaway (flat ~×10 boost on all strong Fe lines = a pure
intercept shift of +ln 10 ≈ 2.3). Structural, not a tuning problem; 50/50 damping does not help because
the loop's fixed point itself is shifted.

**(d) Fix.** Gate per-line correction on an observed signature: (i) measured-vs-expected width ratio
(El Sherbini 2005 — Stark widths are in the DB), or (ii) doublet ratios (implemented, unwired), or
(iii) IRSAC normalization within each species (Sun & Yu 2009 — needs no n·L estimate at all, the right
choice given the solver's admitted order-of-magnitude column-density uncertainty). Treat plasma-state τ
only as a prior, never the sole driver.

---

## F5 — No spectral-response (radiometric) correction anywhere in the inversion intensity path

**Impact: high for the user's own ps instrument; latent for ChemCam (CCS arrives calibrated).**

**(a) Code behavior.** `LineObservation.intensity` is the raw integrated area from the measured spectrum
(`cflibs/inversion/identify/line_detection.py:367-383,425-447`). There is no division by a
detection-efficiency curve anywhere in preprocess/identify/solve; the instrument `response_curve` exists
only in the *forward* model (`cflibs/instrument/model.py`, `cflibs/instrument/kernels.py:41-58`). The
Boltzmann ordinate `ln(Iλ/gA)` is otherwise correct (λ-multiplication present, `data_structures.py:63`).

**(b) Literature requirement.** CF-LIBS is built on *relative line intensities across wide wavelength
spans*; correction for the spectral efficiency of the collection+spectrometer+detector chain is mandatory
(Ciucci et al. 1999; Tognoni et al. 2010 §2 list spectral-efficiency calibration among the experimental
prerequisites; the *Front. Phys.* 2022 CF-LIBS review, doi:10.3389/fphy.2022.887171, devotes a section to
it — deuterium/halogen standards).

**(c) Quantified impact.** A factor R in relative response between one element's lines and another's
enters as δq = ln R between intercepts → C ratio biased by R. Typical uncorrected UV-vs-VIS
grating+detector response differences are ×2–×10 → ln R ≈ 0.7–2.3, the same magnitude as the entire Fe
intercept anomaly. Elements whose lines cluster in different windows (Si/Mg UV vs Na/K VIS) are worst-hit.

**(d) Fix.** Add a response-curve hook (wavelength → efficiency) applied to `intensity` and
`intensity_uncertainty` at observation-build time; let `analyze`/`invert` accept a calibration file;
document that ChemCam CCS input is already radiometrically calibrated (hook defaults to identity).

---

## F6 — Intensity extraction: `max(area, peak_height)` floor mixes incompatible quantities

**Impact: medium-high (systematic ln-space distortion of the Boltzmann ordinate).**

**(a) Code behavior.** Trapezoid path: `line_area = max(line_area, segment_intensity.max())`
(`line_detection.py:430-431`) — replaces an integrated area (counts·nm) by a peak height (counts) whenever
the line is narrower than ~1 nm equivalent. The Voigt-deconvolution path returns a true fitted area with
no such floor (`line_detection.py:367-372`), and the two builders are mixed *within one spectrum*
(`_build_observation_for_match`, `line_detection.py:633-662` — Voigt when a fit exists for that peak,
trapezoid otherwise).

**(b) Literature requirement.** Boltzmann/Saha–Boltzmann ordinates require the wavelength-integrated line
intensity for every line on a common scale (Aguilera & Aragón 2007; Tognoni et al. 2010).

**(c) Quantified impact.** For two lines treated by different rules, the ordinate offset is
|ln(FWHM_eff/1 nm)| ≈ 1.5–3 ln-units at typical 0.05–0.3 nm widths — larger than the entire physical
q_Si−q_Fe spread (~3) the closure is trying to measure. Whether a line gets Voigt or trapezoid treatment
depends on deconvolution success, i.e. effectively on SNR — a bias correlated with brightness.

**(d) Fix.** Delete the `max()` floor (keep a positivity check); on deconvolution failure fall back to
trapezoid area only; never mix peak heights and areas. Unit-test that the two builders agree to ~10% on a
synthetic isolated Gaussian.

---

## F7 — Boltzmann weighting: w ∝ I single-line dominance; the SB-graph gain is *only* the weighting change

**Impact: medium-high (partially mitigated by `boltzmann_weight_cap`; misattribution blocks the real fix).**

**(a) Code behavior.** σ_y = σ_I/I with the Poisson floor σ_I ≈ √(I·Δλ) (`line_detection.py:300-331`)
gives w = 1/σ_y² ∝ I: the brightest line owns its element's intercept (documented in-repo: Fe I 382 nm
carried 133× the next Fe weight, lifting q_Fe by +2.2 → e^2.2 ≈ 9× in C_Fe, `iterative.py:931-958`). Two
mitigations coexist: a per-element cap `w ≤ 5×median` (`_cap_boltzmann_weights`, `iterative.py:2842-2875`)
on the default plane, and the opt-in SB-graph which is **unweighted** (`iterative.py:2707-2746`). This
audit verified numerically (Frisch–Waugh–Lovell) that the per-element-centered pooled fit
(`_fit_common_boltzmann_plane`, `iterative.py:1398-1501`) and the global dummy-variable lstsq
(`_solve_sb_graph_lstsq`, `iterative.py:2749-2817`) produce **identical slopes and intercepts to 1e-14
under equal weights**. The in-code claim that the SB-graph's "global geometry conditions the intercepts"
(`iterative.py:960-984`) is misattributed: with the same line set and weights they are the same estimator.
The measured RMSE gain comes from (i) unit weights and (ii) the SB-graph retaining single-line elements
that the common-slope path silently drops (`iterative.py:1421-1423` requires ≥2 lines/element; the graph
only needs the global system determined, `iterative.py:2775`).

**(b) Literature requirement.** Aguilera & Aragón 2007 construct the multi-element (Saha-)Boltzmann plot
from *calibrated emissivities* with uniform treatment of lines; standard WLS (Bevington & Robinson,
*Data Reduction and Error Analysis*, §6) is only optimal when the σ_y model is *correct* — Poisson-only σ
on 4-decade intensities ignores the dominant per-line systematics (A_ki grades 10–50%, self-absorption,
blends, response), so the implied weights are wrong by orders of magnitude precisely for bright lines.

**(c) Quantified impact.** Documented in-repo: weighted SB-graph RMSE 14 vs unweighted 8 (ChemCam BHVO-2);
default plane Fe → 72% without the cap.

**(d) Fix.** Replace w = 1/σ_stat² with a total-error model σ_y² = (σ_I/I)² + σ_Aki² + σ_floor², with
σ_floor ≈ 0.2–0.5 ln-units for un-modeled per-line systematics (this bounds any line's weight naturally and
removes both the ad-hoc cap and the all-or-nothing unweighted choice). The `aki_uncertainty` folding
already implemented (`cflibs/inversion/physics/boltzmann.py:896-931`) is the right pattern — extend it
with the systematic floor and apply it identically on *both* fit paths.

---

## F8 — `converged=True` on degenerate solves: the lax path lacks gates and the metrics

**Impact: medium (silently green-lights garbage).**

**(a) Code behavior.** The Python path gates convergence on slope sign, R² ≥ `min_boltzmann_r2` (default
0.3), and the keystone-collapse test (`iterative.py:1917-1919,1989-2002,2157-2161`) and always emits
`boltzmann_r_squared` (`iterative.py:2048`). The **lax path** (`_solve_lax` + `_run_lax_while_loop`)
gates only on slope/R² (`iterative.py:782-784,818-824`): **no closure-degeneracy gate**, and its quality
dict contains only `r_squared_last` + LTE keys (`iterative.py:2298-2299`) — so a consumer reading
`quality_metrics.get("boltzmann_r_squared")` gets `None` while `converged=True`. This is the exact
reported symptom; `IterativeCFLIBSSolverJax.solve` routes there (`iterative.py:3125-3175`). Additionally
convergence is a *step-size* test (|ΔT|<100 K, Δn_e/n_e<0.1 under 50% damping, `iterative.py:1997-2002`) —
a slowly drifting iterate can pass it.

**(b) Literature requirement.** No formal convergence canon exists (Ciucci 1999 is single-pass), but
published quality gates are: Boltzmann R² ≥ 0.8–0.95, inter-element T spread ≤ 10–15%, Saha–Boltzmann T
consistency ≤ 20–30%, closure residual ≤ 5–10% — these exact thresholds are already encoded in
`quality.py:77-83` (per Tognoni et al. 2010; Grifoni et al. 2016, *Spectrochim. Acta B* 124, 40,
doi:10.1016/j.sab.2016.08.022); they are just never enforced. Line-selection canon: E_k span well above kT
(the repo's `min_energy_spread_ev = 2.0` ≈ 2.3 kT at 1 eV is reasonable but is a *warning*, not a gate —
`line_selection.py:286-300`), ≥3–4 lines/element (warning only, `line_selection.py:302-314`). Note the
CLI keeps resonance lines by default (`exclude_resonance=False`, `cli/main.py:319-328`) with SA off — a
known optical-depth exposure no gate catches.

**(d) Fix.** Port the three Python gates + full quality dict to the lax assembly (all inputs are in
`LoopState`); make `QualityAssessor` thresholds part of the `converged` definition (or a tri-state
`quality_flag` consumed by the CLI); add a fixed-point residual test (re-evaluate one undamped iteration
at the final state) instead of the damped step-size test.

---

## F9 — Uncertainty reporting is structurally right but quantitatively dishonest

**Impact: medium.**

**(a) Code behavior.** `solve()` returns zeros for all uncertainties (`iterative.py:2170-2182`).
`solve_with_uncertainty` does propagate slope/intercept correlation through the closure with a shared
`slope_u` (`iterative.py:2360-2392`; `cflibs/inversion/physics/uncertainty.py:232-282`) — the correlation
bookkeeping matches the Tognoni-2010 error-budget structure. But: (i) `y_mean_err = √(1/Σw)`
(`iterative.py:2379`) assumes the stated σ_y are correct and ignores residual scatter — no χ²/dof
inflation, so when the plane misfits (every real-rock case) σ_C is underestimated by the misfit factor;
(ii) on the SB-graph path weights are unit, so `√(1/Σw) = 1/√N` is dimensionally meaningless;
(iii) U_s(T) and the Saha exp(−χ/T) are treated as exact in closure propagation (T variance enters only
via `temperature_from_slope`, `uncertainty.py:160-179`, and is *not* fed into M_s or U_s — the dominant
non-linear terms of F3); (iv) n_e variance enters only if the caller passes `n_e_relative_uncertainty`
(default 0).

**(b) Literature requirement.** Tognoni et al. 2010 §4: concentration uncertainty must include T-error
propagation through both exp(E_k/kT) *and* U(T) and the Saha terms; a-posteriori (scatter-based) error
scaling is standard WLS practice when χ²/dof ≫ 1.

**(d) Fix.** Scale intercept covariances by max(1, χ²/dof) per element; propagate σ_T into U_s(T)
numerically (finite difference — providers are cheap); document `MonteCarloUQ` (`uncertainty.py:790+`) as
the honest path and wire it into the CLI `--uncertainty` mode for real samples.

---

## F10 — Loop architecture: the fixed-point iteration is workable but the joint fit should be promoted

**Impact: medium (robustness; the pieces already exist in-repo).**

**(a) Code behavior.** T comes from the pooled all-element common slope (good — this *is* the
multi-element SB plot of Aguilera & Aragón 2007), with 50% damping on T and n_e
(`iterative.py:1924,1981`) and the R²/slope-sign T-guard. The loop is a Gauss–Seidel fixed point over
(T, n_e, C, SA); its failure modes (F4 feedback, step-size convergence, mid-iteration U(T) lag) are all
artifacts of the splitting. A joint estimator already exists twice in-repo: `JointOptimizer` (L-BFGS over
(T, log n_e, C) against a forward model, `cflibs/inversion/solve/joint_optimizer.py:229+`) and the
Bayesian forward model (full-spectrum, NumPyro NUTS, `cflibs/inversion/solve/bayesian/forward.py:98+`),
plus `ClosedFormILRSolver` (single WLS with slope + ILR coordinates + shared intercept,
`cflibs/inversion/solve/closed_form.py:220-331` — well-posed, honest covariance from the normal
equations).

**(b) Literature.** Grifoni et al. 2016 compare iterative CF-LIBS against C-sigma and
fundamental-parameter (synthetic-spectrum-fit) approaches and find the global-fit families more robust to
line-level pathologies; Aragón & Aguilera's Cσ graphs (*J. Quant. Spectrosc. Radiat. Transf.* 149 (2014)
90, doi:10.1016/j.jqsrt.2014.07.026) fold curve-of-growth self-absorption *into* the regression instead
of pre-correcting intensities — directly the cure for F4.

**(d) Recommendation.** Keep the iterative solver as initializer; promote
`ClosedFormILRSolver`/`JointOptimizer` (seeded by it, with Stark-n_e fixed per F2) as the production
estimator; evaluate a Cσ-graph mode (one weighted regression over all lines with per-line COG factor) as
the principled SA treatment. All are physics-only (no ML-constraint violation).

---

## Lower-priority observations

1. **Saha shift form is correct** (Q2): `x* = E_k + χ_eff`, `y* = y − ln[(2(2πm_ek)^{3/2}/h³)(T^{3/2}/n_e)]`
   with `SAHA_CONST_CM3 = 6.042e21` verified to 0.1% (computed 6.037e21 cm⁻³ eV^−1.5); U_II/U_I correctly
   *omitted* from the shift (absorbed into the fitted neutral intercept). IPD is implemented consistently
   in both the shift and the multipliers (`iterative.py:1216-1229`) but **off by default**
   (`apply_ipd=False`, `iterative.py:863`); Debye–Hückel Δχ ≈ 0.06 eV at (1 eV, 1e17) → ~6% Saha effect —
   enable it.
2. **Dead API:** the `closure: ClosureStrategy` constructor arg is stored (`iterative.py:1060-1064`) but
   never consumed by `solve` — closure selection is the `closure_mode` string. Remove or wire it.
3. **Corona empiricism:** `T_saha = 0.3·T + 0.7·T_corona` for {Si,Fe,Ca,Al,Mg} and `T_corona = 0.8·T`
   (`iterative.py:1146-1155,1926-1931`) is honestly documented as non-literature; it should not survive an
   overhaul (two-zone physics belongs in `solve/bayesian/two_zone.py`).
4. **Oxide closure** is a correct molar-oxygen balance with fixed geological oxidation states
   (`closure.py:39-98`); ILR/PWLR closures are single-pass-equivalent to standard normalization by
   construction (`closure.py:719-880`) — they change conditioning, not physics.
5. **τ estimate uses g_i ≈ g_k** (`self_absorption.py:1159`, documented O(1) bias) — plumb true g_i via
   `CDSBLineObservation`, which already carries it (`cflibs/inversion/physics/cdsb.py:63-91`).
6. **Multiplet aggregation** (Wakil-style gA summing, `boltzmann.py:291-324`) and NIST-grade A_ki
   weighting are good practice — keep.

---

## Answers to the audit questions (cross-reference)

| Q | Verdict | Findings |
|---|---|---|
| 1 | Ordinate ln(Iλ/gA) correct; **no spectral-response correction**; w∝I weighting pathological, cap is a band-aid; common-slope fit = Aguilera–Aragón multi-element SB plot (algebra verified) | F5, F7, F6 |
| 2 | Saha shift correct incl. optional IPD (off by default); Saha constant verified; pooled SB-graph = same estimator as the default plane under equal weights | F7, obs. 1 |
| 3 | **Element total = n_I + n_II IS used** (M_s = 1+S everywhere). Real defects: U_II(Na)=15 fallback (~15×), invented n_e in S, U_s(T) amplification of T bias | F1, F2, F3 |
| 4 | Escape-factor COG divide (Bulajic-flavored), τ from recovered composition, runs every iteration; positive feedback explains "SA makes inflation worse"; doublet tools exist unwired; no IRSAC/CD-SB | F4 |
| 5 | Stark machinery exists, never wired by the CLI; default = 1-atm pressure balance (physically invalid, self-acknowledged) | F2 |
| 6 | Python gates OK (slope sign, R²≥0.3, keystone); lax path lacks keystone gate + metrics → `converged=True` / `boltzmann_r_squared=None`; published quality thresholds encoded but unenforced; E_k-span (2 eV ≈ 2.3 kT) and min-lines checks are warnings only | F8 |
| 7 | Correlated slope/intercept propagation exists, but no χ²/dof scaling, SB-graph weights meaningless, U(T)/Saha treated exact → underestimates | F9 |
| 8 | T from all-element common slope (correct), 50% damping, R²-gated; loop is a fixable Gauss–Seidel splitting; promote in-repo joint/closed-form fits, consider Cσ graphs | F10, F3 |

## References

- Ciucci, Corsi, Palleschi, Rastelli, Salvetti, Tognoni (1999). *Appl. Spectrosc.* 53, 960–964. doi:10.1366/0003702991947612.
- Tognoni, Cristoforetti, Legnaioli, Palleschi (2010). *Spectrochim. Acta B* 65, 1–14. doi:10.1016/j.sab.2009.11.006.
- Aguilera & Aragón (2007). *Spectrochim. Acta B* 62, 378–385. doi:10.1016/j.sab.2007.03.024.
- Aragón & Aguilera (2008). *Spectrochim. Acta B* 63, 893–916. doi:10.1016/j.sab.2008.05.010.
- Aragón & Aguilera (2014). *J. Quant. Spectrosc. Radiat. Transf.* 149, 90–102. doi:10.1016/j.jqsrt.2014.07.026.
- Bulajic et al. (2002). *Spectrochim. Acta B* 57, 339–353. doi:10.1016/S0584-8547(01)00398-6.
- Sun & Yu (2009). *Talanta* 79, 388–395. doi:10.1016/j.talanta.2009.03.066.
- El Sherbini et al. (2005). *Spectrochim. Acta B* 60, 1573–1579. doi:10.1016/j.sab.2005.10.011.
- Cristoforetti & Tognoni (2013). *Spectrochim. Acta B* 79–80, 63–71. doi:10.1016/j.sab.2012.11.010.
- Cristoforetti et al. (2010). *Spectrochim. Acta B* 65, 86–95. doi:10.1016/j.sab.2009.11.005.
- Cavalcanti et al. (2013). *Spectrochim. Acta B* 87, 51–56. doi:10.1016/j.sab.2013.05.016.
- Grifoni et al. (2016). *Spectrochim. Acta B* 124, 40–46. doi:10.1016/j.sab.2016.08.022.
- CF-LIBS review (2022). *Front. Phys.* 10:887171. doi:10.3389/fphy.2022.887171.
