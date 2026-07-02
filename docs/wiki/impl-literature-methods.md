---
slug: impl-literature-methods
title: "Code: Implementations of Literature Methods (incl. Forward Model)"
chapter: impl-literature-methods
order: 10
status: stable
register: code-walkthrough
summary: >
  Code-anchored walkthrough of every shipped LITERATURE method: the forward model
  (SpectrumModel → SingleZoneLTEPlasma → SahaBoltzmannSolver → emissivity → profiles →
  instrument) and the inversion methods (preprocess/RANSAC, the six paper-faithful
  identifiers, Boltzmann fit, closure variants, self-absorption, Stark n_e, the
  IterativeCFLIBSSolver loop, closed-form ILR, C-sigma, Bayesian NUTS+dynesty, joint
  L-BFGS-B). Each entry cross-links its verified bug ledger and its literature citation.
tags: [forward-model, saha-boltzmann, emissivity, voigt, ransac, identifiers, closure, self-absorption, stark, iterative-solver, bayesian, joint-optimizer, code-walkthrough]
updated: 2026-07-02
benchmarks_pre_reset: true
sources:
  - "@ciucci1999"
  - "@aragon2008"
  - "@tognoni2010"
  - "@aguilera2007"
  - "@cristoforetti2010"
  - "@hermann2017"
  - "@stewart1966"
  - "@barklem2016"
  - "@irwin1981"
  - "@olivero1977"
  - "@zaghloul2011"
  - "@labutin2013"
  - "@gajarska2024"
  - "@noel2025"
  - "@miller2018"
  - "@badiu2017"
  - "@fischler1981"
  - "@rodgers2000"
  - "@hoffman2014"
  - "@speagle2020"
  - "@cash1979"
  - "@egozcue2003"
  - "@aragon2014"
  - docs/v4/overhaul/verified/plasma.md
  - docs/v4/overhaul/verified/radiation.md
  - docs/v4/overhaul/verified/instrument.md
  - docs/v4/overhaul/verified/inv-preprocess.md
  - docs/v4/overhaul/verified/inv-identify.md
  - docs/v4/overhaul/verified/inv-physics.md
  - docs/v4/overhaul/verified/inv-solve.md
  - docs/v4/overhaul/verified/inv-common.md
  - docs/M5-pipeline-sensitivity-study.md
  - docs/M-spec-lever-comparison.md
  - cflibs/radiation/spectrum_model.py
  - cflibs/plasma/saha_boltzmann.py
  - cflibs/inversion/solve/iterative.py
code_refs:
  - cflibs/radiation/spectrum_model.py::SpectrumModel
  - cflibs/radiation/emissivity.py::calculate_line_emissivity
  - cflibs/radiation/profiles.py
  - cflibs/radiation/kernels.py
  - cflibs/plasma/state.py::SingleZoneLTEPlasma
  - cflibs/plasma/saha_boltzmann.py::SahaBoltzmannSolver
  - cflibs/plasma/partition.py
  - cflibs/instrument/model.py::InstrumentModel
  - cflibs/inversion/preprocess/wavelength_calibration.py
  - cflibs/inversion/preprocess/preprocessing.py
  - cflibs/inversion/identify/alias.py
  - cflibs/inversion/identify/comb.py
  - cflibs/inversion/identify/correlation.py
  - cflibs/inversion/identify/spectral_nnls.py
  - cflibs/inversion/identify/hybrid.py
  - cflibs/inversion/identify/model_selection.py
  - cflibs/inversion/physics/boltzmann.py::BoltzmannPlotFitter
  - cflibs/inversion/physics/closure.py::ClosureEquation
  - cflibs/inversion/physics/self_absorption_observable.py
  - cflibs/inversion/physics/stark_ne.py
  - cflibs/inversion/physics/csigma.py
  - cflibs/inversion/solve/iterative.py::IterativeCFLIBSSolver
  - cflibs/inversion/solve/closed_form.py::ClosedFormILRSolver
  - cflibs/inversion/solve/joint_optimizer.py
  - cflibs/inversion/solve/bayesian
lean_refs:
  - CflibsFormal/Boltzmann.lean#boltzmann_plot
related: [libs-physics, cf-libs-family, classical-quantification, architecture, impl-novel-techniques, error-budget-and-falsification, atomic-data-and-datasets]
supersedes:
  - docs/API_Reference.md
  - docs/CF-LIBS_Codebase_Technical_Documentation.md
---

*[Wiki home](index.md) · [Glossary](glossary.md) · [Bibliography](bibliography.md)*

# Code: Implementations of Literature Methods (incl. Forward Model)

This chapter is the code-map spine for every **published** method CF-LIBS ships: the
forward model that turns `(T, n_e, composition)` into a synthetic spectrum, and the
inversion methods that run it backwards. Each entry names the canonical symbol/path,
states the physics or theorem it realises, cites the source paper, and links the
adversarially-verified bug ledger that says *what is actually wrong here*.

> [!IMPORTANT] PRE-RESET NUMBERS - the atomic DB was rebuilt (ASD59, 203k lines); the composition/F1/RMSE figures quoted from `docs/M5-pipeline-sensitivity-study.md` and `docs/M-spec-lever-comparison.md` are NOT current. Mechanism retained; treat magnitudes as dead. Sensitivity *rankings* (which lever bites) survive the reset; absolute wt% do not.

**Wavelength convention.** The atomic DB stores **air** wavelengths (NIST/ASD convention);
the single conversion utility lives in `cflibs/core/`. Every wavelength in this page is in
nm and is air unless a symbol is flagged vacuum. Notation symbols ($T$, $n_e$, $A_{ki}$,
$g_k$, $E_k$, $U_s$, $\tau$, $C_s$, $F$) are defined once in
[formal-spec/notation.md](formal-spec.md) and never redefined here.

**How to read the ledger links.** The `docs/v4/overhaul/verified/*.md` files are frozen,
file:line, adversarially-verified finding lists (with FALSE-positive call-outs preserved).
They are linked below as the authoritative "known defects" appendix. A finding marked
**FALSE** in those files is a census claim that verification *rejected* — do not act on it.

---

## 1. The forward model {#forward-model}

The forward model is one composition: `SpectrumModel` orchestrates a `SahaBoltzmannSolver`
over a `SingleZoneLTEPlasma`, converts level populations to emissivity, broadens each line,
and folds in the instrument. It is the classic single-zone LTE emission model of
[@ciucci1999] and the review formalism of [@aragon2008] / [@tognoni2010].

### 1.1 `SpectrumModel` — the orchestrator {#spectrum-model}

`cflibs/radiation/spectrum_model.py:44` (`SpectrumModel`), entry point
`compute_spectrum()` at `spectrum_model.py:291`.

| Stage | Code | Physics |
|-------|------|---------|
| Ionization + excitation | `solver.solve_species_states` (`plasma/saha_boltzmann.py:502`) | Saha + Boltzmann |
| Level → emissivity | `calculate_line_emissivity` (`radiation/emissivity.py:16`) | $\varepsilon = \frac{hc}{4\pi\lambda} A_{ki} n_k$ |
| Line broadening | `profiles.py` / `kernels.py` | Gaussian / Voigt |
| Instrument | `_apply_downstream_convolution` (`spectrum_model.py:268`) | fixed-FWHM or $R$ |

The default broadening mode is `BroadeningMode.LEGACY` (`spectrum_model.py:65`). This is a
**non-physical magic number**: `sigma = 0.01 * sqrt(T_eV / 0.86)` (`kernels.py:710,1093`),
a fixed 0.01 nm reference and a 0.86 eV ≈ 10 000 K scale with no wavelength dependence over
a 200–900 nm band. `PHYSICAL_DOPPLER` and `NIST_PARITY` are the physical modes; LEGACY emits
no `DeprecationWarning`. See `radiation.md` Finding 2.

> [!WARNING] BENCHMARK-GATED - two population code paths must stay in sync (`radiation.md` Finding 3). `SpectrumModel.compute_spectrum` computes populations via `solver.solve_species_states` (detailed direct-sum $U$ with IPD) and injects them through `_precomputed_n_upper_per_line`; the JAX kernel's own path uses polynomial partition functions via `_saha_three_stage_populations` (`kernels.py:859-865`). They diverge at high $n_e$; there is no parity regression test. Any physics change must be applied to both.

Additional confirmed issues: the `min_relative_intensity` threshold silently differs by mode
(0.01 for `NIST_PARITY`, 10.0 otherwise, `spectrum_model.py:326`) with no rationale, and the
resolving-power downstream convolution uses the *midpoint* wavelength as a scalar $\sigma$
proxy — a 3× error across 250–750 nm at $R=1000$ (`radiation.md` Finding 4).

### 1.2 `SingleZoneLTEPlasma` + composition conversions {#plasma-state}

`cflibs/plasma/state.py:223` (`SingleZoneLTEPlasma`), a `PlasmaState` subclass. The
load-bearing content is the composition algebra that lets closure operate in whichever space
the physics needs:

| Function | `state.py` | Converts |
|----------|-----------|----------|
| `mass_fractions_to_number_fractions` | 108 | wt% → mol fraction |
| `number_fractions_to_species_densities` | 80 | mol fraction → $N_s$ (cm⁻³) |
| `mass_fractions_to_species_densities` | 147 | wt% → $N_s$ |
| `species_densities_to_number_fractions` | 63 | $N_s$ → mol fraction |
| `from_number_fractions` / `from_mass_fractions` | 265 / 305 | constructors |

> [!CAUTION] DEAD-END / DO-NOT - `TwoRegionPlasma` (`state.py:390`) has **zero production callers**. (An earlier `ConcretizationTypeError` concern is now FIXED: the `__init__` guard at `state.py:436` correctly calls `_is_jax_tracer_or_array(T_core)`, so a concrete JAX-array temperature no longer raises.) No observed failure site (`plasma.md` F5). Do not build on the two-region path without a concrete use case.

### 1.3 `SahaBoltzmannSolver` — ionization/excitation + IPD {#saha-boltzmann}

`cflibs/plasma/saha_boltzmann.py:140` (`SahaBoltzmannSolver`). It solves the Saha ratio for
the neutral/singly/doubly ionized ladder and the Boltzmann level population per stage:

$$\frac{N_{s+1}\,n_e}{N_s} = \frac{2 U_{s+1}(T)}{U_s(T)}\left(\frac{2\pi m_e k_B T}{h^2}\right)^{3/2} \exp\!\left(-\frac{\chi_s - \Delta\chi}{k_B T}\right)$$

$$n_k = N_s \frac{g_k}{U_s(T)} \exp\!\left(-\frac{E_k}{k_B T}\right) \quad (\text{lean:}\texttt{CflibsFormal/Boltzmann.lean\#boltzmann\_plot})$$

Ionization-potential depression $\Delta\chi$ is Debye–Hückel (`DebyeHuckelIPD`,
`saha_boltzmann.py:30`), the Stewart–Pyatt family of [@stewart1966]. The canonical value is
$\Delta\chi_{DH}\approx 0.066$ eV at $n_e = 10^{17}$ cm⁻³, $T = 10^4$ K.

> [!NOTE] The **partition-function cutoff must share the Saha IPD** {#pf-ipd-invariant}: the level sum in $U_s(T)$ must be truncated at the same IPD-lowered ionization limit used in the Saha exponent, otherwise the ion/neutral ratio and the level normalisation use inconsistent continuum edges. `partition.py` accepts an IPD argument for exactly this. This self-consistency requirement is the Barklem & Collet [@barklem2016] cutoff-aware discipline; the polynomial form $\log U = \sum_n a_n (\log T)^n$ used in the vmap'd path is Irwin [@irwin1981].

Known defects (`plasma.md`): the `ionization_potential_lowering` doctest asserts a wrong upper
bound `<= 0.06` below the actual 0.066 eV (F1); the `@cached_partition_function` key pickles
`self` on every miss (overhead + fragility, not the claimed cross-instance miss — F2 was
**downgraded**); and the pure-Python `_direct_sum_energy_levels` loop (F3) is **FALSE as a
hotspot** — production returns early via `partition_function_for` → `provider.at()`.

### 1.4 The canonical emissivity function {#emissivity}

`cflibs/radiation/emissivity.py:16` (`calculate_line_emissivity`):

$$\varepsilon_\lambda = \frac{hc}{4\pi\lambda}\,A_{ki}\,n_k$$

The function returns **line-integrated** emissivity (W m⁻³ sr⁻¹); spectral emissivity is
obtained by multiplying by a normalised profile $\phi_\lambda(\lambda)$ downstream. Populations
above the IPD cutoff return $n_k = 0$ (`upper_level_population_cm3`, `emissivity.py:70`),
computed from the transition's own $(g_k, E_k)$ — no float-keyed join against the
`energy_levels` table (audit F1, bead z3cg). This is the emission coefficient of every LIBS
forward model [@aragon2008].

### 1.5 Voigt / Gaussian / Lorentz profiles {#profiles}

`cflibs/radiation/profiles.py`. The Voigt is the convolution of a Doppler Gaussian and a
Stark/collisional Lorentzian; the NumPy path uses `scipy.special.wofz` (the Faddeeva function,
[@zaghloul2011]) with a pseudo-Voigt fallback when scipy is absent.

> [!WARNING] BENCHMARK-GATED - the pseudo-Voigt total-width constants are a **transcription error**. Code uses `0.5346` and `0.2166` (`profiles.py:257`); Olivero & Longbothum [@olivero1977] give `0.5343` and `0.2169`. The deviation (~0.03% in $f_V$) is sub-noise for LIBS but is a genuine mis-citation, and it is duplicated in **three** files: `radiation/profiles.py`, `radiation/stark.py:200-201`, and `jitpipe/stark.py:79-80` (the on-device Stark $n_e$ path). Fix all three together (`radiation.md` Finding 1 + Missed A).

The NumPy per-line Gaussian is an unvectorised Python loop (`profiles.py:154-157`,
`336-340`) — the CPU real-data path (`radiation.md` Finding 5).

### 1.6 `radiation/kernels.py` — the jittable path, parity-tested {#kernels}

`cflibs/radiation/kernels.py` holds the `@jit_if_available` JAX kernels that mirror the NumPy
forward model for the manifold and Bayesian consumers. The Voigt normalisation is tested
(`tests/test_profiles_jax.py::test_voigt_normalization`, integral ≈ 1 to `<1e-3`), but the
**loop-vs-kernel Gaussian parity** and the **two-population-path parity** (§1.1) are *not*
directly regression-tested (`radiation.md` Findings 3, 7).

### 1.7 Instrument model — fixed FWHM vs resolving power {#instrument}

`cflibs/instrument/model.py:17` (`InstrumentModel`). Two modes: fixed
`resolution_fwhm_nm` (`model.py:39`, $\sigma = \text{FWHM}/2.355$) and resolving-power mode
(`model.py:74`, $\sigma(\lambda) = \lambda/(R\cdot 2.355)$).

> [!WARNING] BENCHMARK-GATED - `InstrumentModel.from_file` **silently drops `resolving_power`** (`model.py:108-147`). A YAML with `resolving_power` only raises a confusing `ValueError`; a YAML with `resolution_fwhm_nm: 0.0` + `resolving_power: 10000` silently produces fixed-FWHM mode with $\sigma=0$, then `SpectrumModel` guards `sigma_conv <= 0 → return intensity` and broadening is silently disabled (`instrument.md` F2). Also: the `2.355` constant deviates from the exact $2\sqrt{2\ln 2}=2.35482$ used by the manifold path (`instrument.md` F1, +0.0076%), so a manifold-lookup parity test carries a non-zero systematic offset. `wavelength_calibration` is a **dead pytree field** with no function caller (F4).

---

## 2. Preprocess — the input-quality lever {#preprocess}

`docs/M5-pipeline-sensitivity-study.md` ranks preprocessing #2 among all accuracy levers:
disabling wavelength calibration ~4×'d the bhvo2 error (pre-reset). RANSAC wavelength
calibration is separately **73% of reference inversion cost** (per
`reference_inversion_hotspot_profile`). Accuracy first — the cost is earned.

### 2.1 Peak detection + sub-pixel centroiding {#peaks}

`detect_peaks` (`preprocess/preprocessing.py:525`) wraps `scipy.signal.find_peaks` and
`peak_widths`.

> [!CAUTION] **No sub-pixel centroiding** (`inv-preprocess.md` Finding 2, HIGH): `detect_peaks` returns the wavelength grid value at the *integer* argmax, with no parabolic/Gaussian centroid. At 0.06–0.1 nm/px this is 0.03–0.05 nm RMS per anchor — comparable to the RANSAC inlier tolerance (0.08 nm). The FWHM data from `peak_widths` is already present to support a 3-point parabolic centroid. The `min_intensity_floor` semantics are also inverted (a "floor" that can *lower* the threshold, `preprocessing.py:586-589`, Finding 3), but default 0.0 means it never triggers.

### 2.2 RANSAC wavelength calibration {#ransac}

`preprocess/wavelength_calibration.py` (a 2119-line monolith, `inv-preprocess.md` Finding 5).
The calibration is RANSAC [@fischler1981]: sample minimal peak↔reference pairs, fit a
shift/affine/quadratic dispersion, count inliers within `inlier_tolerance_nm`, keep the
consensus model. The segmented path (`calibrate_wavelength_axis_segmented`) defaults to
`("shift", "affine")` per CCD segment, re-detecting seams first.

| Issue | Location | Severity | Note |
|-------|----------|----------|------|
| Quadratic fit on un-normalised nm | `wavelength_calibration.py:192-195` | HIGH | cond. ~$3\times10^6$; segmented path excludes quadratic by default (Finding 1) |
| `detect_ccd_seams` pure-Python rolling median | `1302-1305` | HIGH | `scipy.ndimage.median_filter` drop-in available (Finding 4) |
| Hough warm-start `CFLIBS_HOUGH_CALIB` default-OFF, undocumented pass criteria | `258,268` | MED | 80 vs 600 RANSAC iters; no RNG dep; promotable after a benchmark gate (Findings 8, 10) |
| Zero tests for flag-gated Hough/early-exit paths | `tests/` | MED | Finding 10 |

> [!WARNING] BENCHMARK-GATED - RANSAC is RNG-coupled and earns roughly +1.4 wt% (pre-reset) on real data, so it cannot simply be cut for latency; any rewrite must be benchmark-gated (repo has regressed identifier/lever changes 3×).

---

## 3. The identifier comparison matrix {#identifiers}

`cflibs/inversion/identify/` ships **six paper-faithful identifiers**. They are deliberately
**different algorithms with different failure modes — not a merge target.** The 2026-06
architecture review explicitly dropped the proposal to unify the ALIAS peak detector with the
others (`reference_arch_review_2026_06_outcomes`): keeping independent paper-faithful
identifiers is the design.

| Identifier | Module | Paper | Core mechanism |
|------------|--------|-------|----------------|
| ALIAS | `alias.py` | Noël 2025 [@noel2025] | multi-coefficient match of acquired vs simplified-plasma-model spectrum; $k_{det}$ scoring |
| Comb | `comb.py` | Gajarska 2024 [@gajarska2024] | element-specific comb matched filter + interference flagging + micro-parameter drift adjust |
| Correlation | `correlation.py` | Labutin 2013 [@labutin2013] | Saha-Boltzmann synthetic spectrum over $(T,n_e)$ grid, maximise Pearson correlation, then attribute peaks |
| spectral_nnls | `spectral_nnls.py` | Miller 2018 [@miller2018] | non-negative least squares unmix vs pure-element library (Lawson–Hanson) |
| hybrid / hybrid_consensus | `hybrid.py`, `hybrid_consensus.py` | (NNLS + ALIAS ensemble) | convex NNLS prefilter gates the ALIAS physics scorer |
| BIC model_selection | `model_selection.py` | Webb et al. 2020 [@webb2020ic]; Badiu 2017 [@badiu2017] | penalised-likelihood / information-criterion line-reality test |

### 3.1 ALIAS detection threshold {#alias-threshold}

ALIAS's `detection_threshold` is the **paper's $C_{th}$ on the $k_{det}$ scale** (0.5 strict /
0.4 recall). It is *not* a legacy confidence-level floor: the older CL floors (0.01–0.10)
meant "accept everything." PR #292 fixed four callers that bypassed it and pinned a guard at
$\geq 0.3$ (`reference_alias_detection_threshold_cth_scale`). Detection is
`detected = k_det > detection_threshold` (`alias.py:1866`), paper-faithful.

> [!CAUTION] The homegrown `N_matched < 3 → CL = 0` gate (`alias.py:4682`) still fires and, via `_gate_subset_relative_cl` (`alias.py:1236-1241`), demotes a genuine 2-line element with $k_{det} > C_{th}$ to `detected = False`. The comment at `alias.py:1863` claiming this gate was "removed" is **wrong** (`inv-identify.md` F2 HIGH; NEW-2). The paper explicitly supports one-line and sparse-line elements. There is no regression test pinning `confidence > 0` for such elements (F9).

The ALIAS Boltzmann-y consistency check is **correct** — both the legacy T-estimator
(`alias.py:2511`) and `BoltzmannPlotFitter` use $y = \ln(I\lambda/(g_k A_{ki}))$ via
`LineObservation.y_value`; the census claim of a y-axis mismatch was **FALSE** (F1). Other
confirmed items: `_compute_ratio_consistency` returns 0.1 not the documented 0.5 (F3);
`P_SNR` is recomputed redundantly inside `_decide` (F4); per-call mutable state
(`_estimated_T`, F5) is documented but latent; `temperature_estimator_mode="weighted"` is a
dead investigation leftover (F8).

**Comb.** Interference analysis (`_analyze_interferences`, `_mark_reciprocal_interference`)
delimits blended regions before they enter Boltzmann fits and **is tested** both positively
and negatively (`inv-identify.md` F10 was **FALSE**). The Comb decision-gate was made
paper-faithful in the 2026 sweep, lifting recall from 0.018 → 0.995 (pre-reset;
`project_id_benchmark_window_ceiling`).

> [!IMPORTANT] PRE-RESET NUMBERS - the complete-DB reset MIS-TUNED the identifiers for a 7× larger catalog (Comb F1 0.529 → 0.329 precision crater; ALIAS recall collapse). Re-tune against the 12-spec `small_v1` corpus, not the ~84-min full ALIAS/RANSAC bench (`reference_complete_db_rebench_baselines`). Make identifiers paper-faithful FIRST; window/aliasing is a secondary precision ceiling.

---

## 4. Boltzmann fit — one physics, three implementations {#boltzmann}

The Boltzmann plot is $y = \ln(I_{ki}\lambda/(g_k A_{ki}))$ vs $x = E_k$, slope $-1/(k_B T)$
(lean:`CflibsFormal/Boltzmann.lean#boltzmann_plot`). The multi-element common-slope extension
is Aguilera & Aragón [@aguilera2007]. Three code realisations must agree:

| Impl | Path | Backend | Role |
|------|------|---------|------|
| `BoltzmannPlotFitter` | `physics/boltzmann.py:42` | NumPy | reference WLS + sigma-clip |
| `boltzmann_jax` | `physics/boltzmann_jax.py` | JAX | vmap/grad, manifold + Bayesian |
| `jitpipe/fit.py` | `jitpipe/fit.py` | JAX (on-device) | segmented real-time |

> [!CAUTION] WLS sigma-clip uses an **unweighted** `np.std` (`boltzmann.py:428-432`, `inv-physics.md` F8): the fit down-weights high-variance points but the outlier threshold is computed from the raw residual distribution, biasing $T$ by ~1–5% in high-dynamic-range cases. The JAX path (`_fit_sigma_clip_jax`) mirrors the same logic. `y_uncertainty` also omits `aki_uncertainty` (`inv-common.md` F3) — the fitter adds it in quadrature separately (`boltzmann.py:_build_sigma_y`), so production is correct but external consumers of `obs.y_uncertainty` underestimate.

The multiplet-aware extension (summed unresolved multiplets with effective $gA$) of Völker &
Gornushkin is a known upgrade path, not yet shipped.

---

## 5. Closure variants + softmax simplex {#closure}

`cflibs/inversion/physics/closure.py` (`ClosureEquation`, `closure.py:793`). Closure enforces
$\sum_s C_s = 1$, cancelling the scalar experimental factor $F$ [@ciucci1999]. Four modes plus
a log-ratio geometry:

| Mode | `closure.py` | Basis |
|------|-------------|-------|
| standard | `ClosureEquation` | direct $\sum C_s = 1$ |
| matrix | `matrix_effects.py` | matrix-effect correction (imports `CFLIBSResult` — layering violation F5) |
| oxide | `default_oxide_stoichiometry:76` | geological oxide closure |
| ILR | `ilr_transform:255`, `ilr_inverse:277` | isometric log-ratio [@egozcue2003] |
| softmax simplex | `softmax_closure.py` | differentiable simplex for joint/Bayesian |

The ILR/CLR machinery (`clr_transform:236`, `ilr_propagate_covariance:299`,
`simplex_covariance_from_ilr:374`, ALR/PLR/PWLR variants) is the compositional-data-analysis
apparatus of Aitchison [@aitchison1982] and Egozcue [@egozcue2003]. It exists because closure
lives on a simplex, not a Euclidean space: for the DED/tracking deliverable, **prefer
log-ratios $\ln(N_i/N_j)$ over closure wt%** — ratios are unaffected by the unknown $F$ and by
elements outside the basis.

> [!NOTE] The composition benchmark was historically BROKEN (O-excluded 52%-sum truth vs mole fractions) and the gating was circular (`reference_benchmark_was_broken_circular`). Physics-correctness (NIST/literature-gated) survives; absolute composition-accuracy claims from before the fix are suspect.

---

## 6. Self-absorption — observable-gated is current {#self-absorption}

Optical depth $\tau$ saturates strong lines; the escape factor is $SA(\tau)=(1-e^{-\tau})/\tau$.
CF-LIBS either avoids thick lines (line selection) or corrects them. The **shipped** corrector
is observable-anchored (`physics/self_absorption_observable.py`, bead 0jvr), not the
composition-derived per-line $\tau$.

> [!CAUTION] FALSIFIED: Composition-derived per-line tau improves accuracy
> - **Claim:** compute $\tau$ from the composition-derived lower-level column density and correct each line for thick-line saturation.
> - **Predicted:** lower held-out RMSEP on ChemCam BHVO-2.
> - **Observed:** RMSEP *increased* — a positive-feedback loop ($\tau$ feeds composition feeds $\tau$).
> - **Verdict:** REJECTED; replaced by the observable-anchored corrector.
> - **Evidence:** `docs/research/physics-first-principles-audit.md` Issue 3 (finding F4); `cflibs/inversion/physics/self_absorption_observable.py`.
> - **Date:** 2026-07-02

The doublet-ratio SA estimator (`self_absorption.py:_ordered_doublet_pair:480`) checks
element, stage, and $|\Delta E_k| < 1$ meV but **not** $g_k$ equality (`inv-physics.md` F3
HIGH): for a true same-upper-level doublet $g_k$ cancels, but two near-degenerate
fine-structure terms with different $J$ within 1 meV would use wrong weights. The
columnar-density CD-SB variant (Cristoforetti & Tognoni [@cdsb2013]) that *exploits*
self-absorption, and the C-sigma curve-of-growth [@aragon2014], are the literature
alternatives (see [cf-libs-family.md](cf-libs-family.md)).

---

## 7. Stark n_e — diagnostic vs forward vs jit {#stark-ne}

Electron density from Stark width. Three code paths: the diagnostic
(`physics/stark_ne.py`), the forward-model broadening (`radiation/stark.py`), and the
on-device jit (`jitpipe/stark.py`). All three carry the Olivero constant (§1.5).

> [!CAUTION] H-alpha Stark n_e uses **linear** scaling, not Gigosos $\sim n_e^{0.7}$ (`stark_ne.py:46-50`, `inv-physics.md` F4 HIGH). `("H", 1, 656.28)` is the first-ranked diagnostic (`PREFERRED_DIAGNOSTIC_LINES:83`) yet the DB stores H-alpha under the same linear-in-$n_e$ convention as every other line, giving 20–50% $n_e$ error at LIBS densities. No Gigosos correction path exists. The ion-dynamics reference is Gigosos et al. [@gigosos2003].

The McWhirter LTE floor ($\delta E$ = resonance-line energy) is the correct convention
(`reference_mcwhirter_delta_e_physics`); the default code path uses `max(E_k)` instead — see
§8 (this is the pipeline's single **critical** physics defect).

---

## 8. The IterativeCFLIBSSolver loop {#iterative}

`cflibs/inversion/solve/iterative.py:` — **4160 LOC**, class `IterativeCFLIBSSolver`, result
`CFLIBSResult` (`iterative.py:116`). The module docstring is one line, so this section is
reconstructed from the class/method docstrings. This is the classic Ciucci–Corsi–Palleschi
CF-LIBS loop [@ciucci1999], multi-element Saha-Boltzmann per Aguilera & Aragón [@aguilera2007]:

```
detect → identify → for each iteration:
  Boltzmann plot per element        → T from common slope
  Saha correction (ionic → neutral) → map ion lines onto the neutral plane
  multi-element common-slope fit    → intercepts ∝ N_s / U_s
  closure (Σ C_s = 1)               → concentrations, F cancels
  charge/pressure balance + Stark   → update n_e
until |ΔT|, |Δn_e| converged
```

The Saha $y$-shift is **mathematically correct**: mapping an ionic line onto the neutral plane
gives $y^*_{ion} = y_{ion} - \ln(\text{SAHA\_CONST}/n_e \cdot T_{eV}^{1.5})$ with $x^* = E_k + \chi$;
the $U_{II}/U_I$ ratio **cancels** (Saha numerator vs Boltzmann denominator). The census's
proposed "fix" to add $\ln(U_{II}/U_I)$ would have *introduced* a bug (`inv-solve.md` F1
**FALSE** — the most consequential correction in the ledger). Do not touch
`_apply_saha_correction` on that basis.

**Critical defect (default path).** The McWhirter LTE gate uses $\delta E = \max(E_k)$ instead
of the resonance-line energy (`lte_validator.py:387`; `iterative.py:2704-2719` leaves
`delta_e_override=None` unless `CFLIBS_MCWHIRTER_RESONANCE_DE` is set). For Fe I, $\max(E_k)\approx$
4–6 eV vs resonance ~2.48 eV — a 2–2.4× overestimate, an ~8–14× too-high $n_e$ floor
($\propto \delta E^3$). The correct behaviour is opt-in; the incorrect one is the default
(`inv-physics.md` F1 critical, F9). The direction is *safe-side* (false McWhirter failures,
not false passes) but it is still wrong.

**Other confirmed loop issues** (`inv-solve.md`):

| # | Issue | Sev |
|---|-------|-----|
| F2 | Lax JAX path does n_e update by **1-atm pressure balance only**; supplying Stark diagnostics silently forces the Python path | high |
| F3 | `two_region` uses `T_corona = 0.8·T_core` — **no literature basis** (code says so), ad-hoc element set `{Si,Fe,Ca,Al,Mg}`; default-off | high |
| F5 | `CFLIBSResult` defined in `iterative.py` but imported by `closed_form.py`, `io/exporters.py`, `jitpipe/pipeline.py` — belongs in `common/` | high |
| F6 | `saha_boltzmann_graph=True` silently disables the lax path; no parity test | high |
| F7/F8 | `solve_with_uncertainty` re-runs Saha + PF queries; `_apply_saha_correction` allocates new `LineObservation` for **all** lines per iteration | medium |

`QualityAssessor` (`physics/quality.py`) is now **wired** into the solver
(`_assess_reliability` def `iterative.py:2611`, import `:2637`, call site `:2792`) despite a stale docstring saying "NOT used" (`inv-physics.md` F6).

---

## 9. Closed-form ILR solver {#closed-form}

`cflibs/inversion/solve/closed_form.py:100` (`ClosedFormILRSolver`, config `:52`). A
non-iterative solver that works directly in ILR coordinates [@egozcue2003]: given fixed $T$,
$n_e$ and the Boltzmann intercepts, it inverts closure in log-ratio space in one linear solve,
producing a composition and its ILR covariance without the iterate-to-convergence loop. It
imports `CFLIBSResult` from `iterative.py` (the F5 coupling). Use it as a fast seed for, or a
cross-check against, the iterative loop.

---

## 10. C-sigma solver {#csigma}

`cflibs/inversion/physics/csigma.py`. The C-sigma (Cσ) graph of Aragón & Aguilera
[@aragon2014] is a generalised curve-of-growth: instead of excluding self-absorbed lines, it
plots a $C\sigma$ ordinate that folds optical depth into a multi-element **common-$\sigma$**
fit (aligned with the repo's common-slope step). It is self-absorption-tolerant by
construction and can even measure transition probabilities. This is the principled
radiative-transfer alternative to the post-hoc SA corrector of §6; see
[cf-libs-family.md](cf-libs-family.md) for where it sits among CF variants.

---

## 11. Bayesian — NUTS + dynesty (prefilter mandatory) {#bayesian}

`cflibs/inversion/solve/bayesian/`. Two engines over the JAX forward model
(`BayesianForwardModel`): NumPyro NUTS [@hoffman2014] for the full posterior over
$(T, n_e, \text{composition}, \text{baseline})$, and dynesty [@speagle2020] for the evidence
$Z$ and multimodal posteriors (line-ID ambiguity, ionization-stage degeneracy — failure modes
the iterative solver cannot express).

> [!IMPORTANT] The NNLS candidate prefilter is **mandatory**, not optional: `select_candidate_elements` (`cflibs/inversion/candidate_prefilter.py`) does an NNLS top-K down-select [@miller2018] before MCMC. Full-element MCMC is intractable. This is an architectural precondition of the Bayesian path.

> [!CAUTION] The Gaussian likelihood default is **Pearson-biased** (`bayesian/likelihood.py:34-82`, `inv-solve.md` F4): putting the model prediction in the variance denominator biases the fitter toward under-estimating peaks. For shot-noise-dominated ICCD LIBS the Poisson (Cash 1979 [@cash1979]) path is correct, but it is off by default "to avoid silently changing existing posteriors." Opt into Poisson for real detector noise.

Calibration of the posterior (SBC rank-uniformity, TARP coverage) is the validation discipline
tracked in [benchmarks-reliability-workflows.md](benchmarks-reliability-workflows.md).

---

## 12. Joint L-BFGS-B optimizer {#joint}

`cflibs/inversion/solve/joint_optimizer.py`. A single non-linear least-squares fit over all
plasma parameters at once, parameterised for unconstrained optimisation:

- temperature as $\log T_{eV}$;
- electron density as $\log_{10} n_e$ (wide dynamic range, `joint_optimizer.py:18`);
- composition as `softmax(theta)` via `softmax_closure` (`joint_optimizer.py:19,56`) — the
  simplex constraint is satisfied by construction.

Minimised with L-BFGS-B (or `jax.scipy.optimize.minimize`, `joint_optimizer.py:54`), it is an
optimal-estimation-style objective in the Rodgers [@rodgers2000] sense; `compute_oe_diagnostics`
(`joint_optimizer.py:127`) exposes averaging-kernel / degrees-of-freedom diagnostics.

> [!NOTE] Register a **CORRECTLY NEUTRAL** stance: the joint solver is a legitimate alternative parameterisation, **not** a proven accuracy win. An earlier "joint beats iterative" headline was contradicted for lack of an adoption gate — the open problem is *when to trust a converged joint fit* (posterior/OE uncertainty is the signal), not the optimiser itself. Do not carry a "joint beats iterative" claim forward without a trust gate. See [error-budget-and-falsification.md](error-budget-and-falsification.md).

---

## 13. What the M5 / M-spec studies teach (mechanism, not magnitude) {#lever-lessons}

> [!IMPORTANT] PRE-RESET NUMBERS - every wt%/RMSE below is pre-ASD59-reset. Mechanism retained; magnitudes dead.

From `docs/M5-pipeline-sensitivity-study.md` (OAT sweeps, two real datasets):

- **`min_lines_per_element` is the #1 fragility.** Tightening past 5 is catastrophic — it
  starves the Boltzmann fits (bhvo2 RMSE 2.43 → 11.35). The best value is 2–3, not higher.
- **Wavelength calibration is the #2 accuracy lever** (an input-quality lever, §2).
- **`max_lines_per_element` is saturated at 20**: 20/30/50/100 are bit-identical; the upstream
  line-strength gate leaves < 20 candidates. Raising it buys nothing; lowering below ~12
  truncates real signal.
- Several gates (`min_snr`, `min_energy_spread_ev`, `peak_width_nm`) are **inert** on these
  spectra — "not falsified," not "validated."

From `docs/M-spec-lever-comparison.md` (formalization-derived selection levers):

- On healthy high-SNR lab data the pipeline is **not selection-constrained** — reliability
  ranking, target-sigma gates, and refuse-to-report are all **no-ops** in the default regime.
- Reliability-ranked line selection (widest upper-level energy spread, the proven $2/|\Delta E|$
  conditioning) **wins only when a cap is forced to bind** (`max_lines=6`): −0.52 wt%, 17/20
  spectra. It is a conditioning-aware tool for **constrained** regimes (handheld, real-time,
  low-SNR), not a lab-data accuracy win.

The transferable lesson: the accuracy bottleneck on healthy data is **atomic-data accuracy +
wavelength calibration + the solver**, not line selection.

---

## Appendix A: verified bug-ledger index {#ledger-index}

Frozen, adversarially-verified file:line finding lists (FALSE-positive call-outs preserved).
These are the authoritative "what is actually wrong here" appendix for this chapter.

| Ledger | Covers | Highest confirmed |
|--------|--------|-------------------|
| [plasma.md](../v4/overhaul/verified/plasma.md) | Saha-Boltzmann, partition, IPD | MEDIUM (F1 doctest); F3 hotspot **FALSE** |
| [radiation.md](../v4/overhaul/verified/radiation.md) | SpectrumModel, emissivity, profiles, kernels | HIGH (Olivero constants ×3) |
| [instrument.md](../v4/overhaul/verified/instrument.md) | InstrumentModel, echelle, pytree | HIGH (F1 2.355, F2 from_file, F4 dead field) |
| [inv-preprocess.md](../v4/overhaul/verified/inv-preprocess.md) | RANSAC, peaks, seams | HIGH (F1,F2,F4,F5) |
| [inv-identify.md](../v4/overhaul/verified/inv-identify.md) | ALIAS/Comb/NNLS | HIGH (F2 CL-zero → detected=False); F1/F6/F10 **FALSE** |
| [inv-physics.md](../v4/overhaul/verified/inv-physics.md) | Boltzmann, SA, Stark, closure | **critical** (F1 McWhirter max(E_k)); F2 **FALSE** |
| [inv-solve.md](../v4/overhaul/verified/inv-solve.md) | iterative, bayesian, joint | HIGH (F2,F3,F5,F6,F10); F1 **FALSE** |
| [inv-common.md](../v4/overhaul/verified/inv-common.md) | data structures, PCA, element_id | HIGH (F5 common→physics cycle, F10 test gap) |

## Appendix B: shipped-history provenance {#provenance}

The jittable inversion pipeline was specified in ADR-0004 (`docs/adr/ADR-0004-jittable-inversion-pipeline.md`)
and promoted in ADR-0005 (`docs/adr/ADR-0005-jitpipe-promotion.md`); instrument calibration
was made a first-class input in ADR-0006. The J0–J12 specs
(`docs/adr/specs/J0..J12`) are the shipped-history record of the on-device path
(J1 preprocess, J2 wavelength calibration, J3 line-matching gates, J4 fit/selection/Boltzmann/closure,
J5 self-absorption, J6 Stark $n_e$, J9 segmented calibration, J10 forward-fitting ID, J11
differentiability, J12 scoreboard). This chapter absorbs `docs/API_Reference.md` and the
implementation sections of `docs/CF-LIBS_Codebase_Technical_Documentation.md`; those retain
tombstone redirects here.

---

## See also

- [libs-physics.md](libs-physics.md) — the Saha-Boltzmann and Boltzmann-plot physics this code realises
- [cf-libs-family.md](cf-libs-family.md) — where C-sigma, CD-SB, OPC sit among CF variants
- [classical-quantification.md](classical-quantification.md) — the Boltzmann-plot quantification lineage
- [impl-novel-techniques.md](impl-novel-techniques.md) — the manifold, jitpipe, and evolution code
- [architecture.md](architecture/index.md) — module map, layering violations, the CFLIBSResult coupling
- [error-budget-and-falsification.md](error-budget-and-falsification.md) — the aggregated falsification ledger (joint-solver neutrality, tau feedback)
- [atomic-data-and-datasets.md](atomic-data-and-datasets.md) — the ASD59 reset that invalidated pre-reset numbers
- [formal-spec.md](formal-spec.md) — notation authority and the Lean `boltzmann_plot` theorem
