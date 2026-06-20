# CF-LIBS Accuracy-First Roadmap (Authoritative)

**Status:** Authoritative synthesis · **Date:** 2026-06-20 · **Owner:** synthesis lead
**Supersedes:** ad-hoc lever notes; consolidates the 6 per-lever analyses + literature packs into one program.
**Standing directive (binding):** ACCURACY, PRECISION, and RELIABILITY of T / n_e / composition extraction take
**strict precedence over latency.** Sub-millisecond inference is the eventual goal but is **deferred** until the
pipeline is scientifically robust. Never trade accuracy for speed. Physics-only constraint holds throughout:
shipped `cflibs/` must not import `sklearn`, `torch`, `tensorflow`, `keras`, `flax`, `equinox`, `transformers`,
`jax.nn`, or `jax.experimental.stax` (ML only in `cflibs/evolution/`). Every accuracy-changing recommendation is
**benchmark-gateable** (flag-off vs flag-on on the scoreboard) — the repo has regressed **three times** on ungated
scoring changes, so non-regression gating is mandatory, not optional.

---

## 1. Thesis & Sequencing

**Thesis.** The binding error in CF-LIBS on this codebase is **the inputs**, not the solver math. The plan-v4
controlled round-trip (workflow `wphrxvuyj`, 2026-06-20, V100S, real ExoJAX forward + real ChemCam/SuperCam) measured
the **inversion-algorithm floor at RMSE 2.9×10⁻⁶ (≈0)** — i.e. when the atomic data and the forward model agree, the
solver recovers truth essentially exactly. The same campaign measured that the **atomic-data line-list mismatch alone
injects RMSE 0.171 mass-fraction** — roughly **five orders of magnitude above the solver floor** and of the same order
as the real-data composition error (ChemCam 0.12–0.17 mass-frac). A third measured term, **noisy weak-emitter
ill-posedness, floors at RMSE ≈0.20** (Na/K/Si dynamic range ≈10⁶). Real-data tests were run **uncalibrated**
(Mode C: fixed-FWHM proxy, no E(λ)), and SuperCam composition RMSE was **20–26 wt%**, 4–10× the 1–3 wt% CF-LIBS norm.

The conclusion is unambiguous and is the spine of this roadmap:

> The solver is already correct to ≈0. The accuracy ceiling is set by **atomic data quality, instrument calibration,
> self-absorption on optically-thick majors, and weak-emitter ill-posedness** — in that rough order. Optimizing the
> solver further (and especially optimizing it for *speed*) cannot move composition RMSE while the inputs dominate by
> five orders of magnitude.

**Sequencing.** Work proceeds in three strictly-ordered phases:

1. **ACCURACY** — fix the inputs (atomic data, calibration, self-absorption, n_e identifiability). This is where the
   0.171 and the 20–26 wt% live. (Levers 1–4 below; milestones M1–M6.)
2. **RELIABILITY / UQ** — make every reported number *honest*: refuse-to-report gates (LTE validity, quality flags),
   calibrated per-element uncertainty, and precision-vs-accuracy separation. This does not lower the *unconditional*
   RMSE — it ensures we never emit a confident wrong number, and it lowers the *conditional* (reported-subset) RMSE.
   (Levers 5–6; milestones M7–M9.)
3. **LATENCY — DEFERRED.** Sub-ms is the eventual product goal, but it is gated on the entire accuracy + reliability
   program being complete and benchmark-locked. The fast structured-GN is presently **~2× *less* accurate** than the
   legacy iterative solver on real data (median 0.151 vs 0.076; §3), so promoting it now would be trading accuracy for
   speed — forbidden by the standing directive. Latency is milestone **M10**, last, and explicitly gated.

**Why latency is deferred (stated explicitly).** The fast GN's speed advantage (~2000×) is real and valuable, but it
is presently bought at an accuracy cost we cannot pay: it cannot identify n_e from a flat prior, modeling
self-absorption inside it rails n_e, and it does not consume the atomic-data uncertainty it already stores. None of
these are latency problems — they are accuracy problems that happen to live in the fast path. We perfect accuracy in
the legacy path first, *then* port the validated physics into the fast path under parity gates. The v4 finding —
algorithm floor ≈0 vs atomic-data 0.171 — is the quantitative justification: there is no accuracy to be gained from
the solver, only from the inputs, so spending engineering on a faster solver before the inputs are fixed optimizes the
wrong term.

---

## 2. The Composition / T / n_e Error Budget

Ranked by share of the **measured** real-data accuracy gap. The two anchor numbers are the v4 controlled
measurements: algorithm floor **2.9×10⁻⁶** and atomic-data mismatch **0.171** RMSE mass-fraction. Shares below are
reconciled to those numbers; where a lever has not yet been isolated with its own flag-off/flag-on experiment, the
share is marked **UNMEASURED** and an isolation experiment is the headline deliverable for that lever.

| # | Lever | Share of measured accuracy gap | Cited / measured expected gain | Basis |
|---|-------|--------------------------------|-------------------------------|-------|
| 1 | **Atomic data quality & gf-grade-aware selection** | **DOMINANT.** Owns the measured **0.171 RMSE** atomic-data term — ~5 orders of magnitude above the solver floor; same order as the real-data RMSE. | Targets the full 0.171 term. Repo-measured: **Kurucz beats NIST by −5 to −6 wt% on line-rich SuperCam** (completeness). Per-line: grade-A (≤3%) vs heuristic-B (10%) vs D/E (50%) is a **3–17× per-line accuracy swing** propagating ~1:1 into species bias. | v4 measured; Ciucci 1999; Tognoni 2010; atomic_data lever |
| 2 | **Instrument calibration E(λ) / LSF / axis (ADR-0006)** | **LARGE, partly UNMEASURED.** Second only to atomic data. Real-data was Mode-C uncalibrated → SuperCam **20–26 wt%** (4–10×). E(λ) does **not** cancel under closure; it rotates (T) and scatters (composition) the Boltzmann plot. | Literature norm for well-calibrated, optically-thin LTE CF-LIBS: **1–3 wt%** (target). E(λ) correction removes the slope-rotation T bias and wavelength-region composition bias; magnitude scales with response variation across a species' line set (10s of % when response spans a steep grating/QE edge). | v4 measured (Mode-C); Tognoni 2010; ADR-0006 §1.2; calibration_matrix lever |
| 3 | **Weak-emitter ill-posedness** | **LARGE.** Owns the measured **~0.20 RMSE noisy floor** (Na/K/Si, ~10⁶ dynamic range). Not solver math — fundamental ill-conditioning. | No point-estimate fix; the honest response is **calibrated large CIs + reliability downgrade** (Lever 6), not a tighter wrong number. Tognoni 2007: even an *ideal* plasma has an atomic-data/line-intensity precision floor. | v4 measured; Tognoni 2007; uncertainty_precision lever |
| 4 | **Self-absorption (optically-thick majors)** | **MODERATE, conditional, UNMEASURED in-repo.** Bounded to the optically-thick major subset (Ca/Na/Mg/Al/Fe). Cannot touch the 0.171 atomic-data floor or weak emitters. Concrete symptom: Al I 877 doublet sits **+3 ln** above the resonance-anchored Boltzmann line. | Literature ceiling **~27% → ~2% rel error** when SA dominates; Bulajic ~10× when SA dominates. Realistic repo gain: **single-digit wt% on SA-heavy spectra**, not transformational; doublet-pass often a no-op on real detected lists. | al-calibration-rootcause audit; Bulajic 2002; self_absorption lever |
| 5 | **Stark n_e identifiability & coverage** | **MODERATE, asymmetric. Headline gain already BANKED.** Fixes one specific failure mode: n_e is unidentifiable from a flat prior (**ne_rel ≈0.3** at ~1e17). Pin/seed already implemented in both solvers — residual is **literature-grade DB coverage** (only **244/28,727 lines = 0.85%** carry `stark_b`). | Stark on isolated literature-grade lines: **~10–20% n_e accuracy** (vs measured ~30% from a flat prior). Composition gain is indirect via Saha; 30%→15% n_e roughly halves the Saha-correction error. Incremental gain = converting silent pressure-balance-fallback solves into Stark-pinned solves (each is the ~2× legacy-vs-GN delta). | v4 measured; Konjevic 2002; Aragon-Aguilera 2008; Gigosos 2014; stark_ne lever |
| 6 | **LTE validity / multi-zone reliability** | **SMALL on current RMSE; OWNS the trust surface.** Controlled floor ≈0 ⇒ single-zone math is not the dominant bias *when LTE holds*. But gradient-rich planetary plasmas emit biased single-T composites **with no flag today**. | No unconditional RMSE drop claimed. Gain = **conditional RMSE** (refuse-to-report below "acceptable" excludes the worst solves) + a reported flag-fraction. Thomson-scattering: apparent single-T can diverge from true T_e by **tens of %** in early/non-LTE conditions, propagating to factor-level concentration error. | v4 framing; Cristoforetti 2010; Aguilera 2008; Liu 2015; lte_reliability lever |
| 7 | **Uncertainty calibration & reporting** | **~0% of the point-estimate gap; ~100% of the trustworthiness gap.** UQ never moves the mean — it makes the error bars honest. | Bring classical-path 95%-CI empirical coverage into **[0.93, 0.97]** (the repo's own protocol band, already used for Bayesian). Weak emitters get CIs that bracket truth / a reliability downgrade instead of false "good". | uncertainty_precision lever; posterior_metrics.py band; Tognoni 2007; Safi 2019 |

**Reconciliation to v4.** Levers 1+2+3 account for essentially all of the measured real-data RMSE: atomic data
(0.171) is the dominant term; calibration E(λ) is the second uncontrolled input (Mode-C ⇒ 20–26 wt% SuperCam); weak
emitters floor the noisy term at ~0.20. Levers 4 and 5 are corrections to specific *subsets* (optically-thick majors;
n_e-degenerate spectra) and cannot exceed those subsets. Levers 6 and 7 are reliability/reporting, not RMSE-of-the-mean.
This is internally consistent with algorithm-floor ≈0: every remaining error is an input or an ill-posedness, never the
solver.

---

## 3. The Architectural Fork — Perfect the Legacy Solver First

**Recommendation: YES — perfect the legacy iterative solver first; treat the fast structured-GN as a later latency
play, ported under parity gates only after the accuracy program is complete.**

**Justification from the measured legacy-vs-GN data.** Plan v4 measured the legacy iterative solver
(`cflibs.inversion.pipeline.run_pipeline`, geological preset, `stark_ne=True`) at **median composition RMSE 0.076 vs
the fast structured-GN at 0.151 on real data — ~2× more accurate on 4/5 real spectra** — at the cost of being ~2000×
slower. The standing directive forbids trading that accuracy for the speed. Three measured mechanisms make the legacy
path the correct accuracy host today:

1. **Stark n_e exploitation.** The legacy solver consumes `measure_stark_ne` as its primary per-iteration n_e update
   (50/50 damped); this is *why* it is 2× more accurate — it is the path where the Stark seed is fully used. The fast
   GN merely pins n_e to the Stark value and otherwise cannot identify n_e from a flat prior (ne_rel ≈0.3).
2. **Self-absorption placement.** Modeling SA inside the fast GN **rails n_e** (v4 failure mode). Observable-gated SA
   correction belongs **upstream** in the legacy path (it is already correctly wired there, applied once before the
   Boltzmann fit). The fast GN is structurally the wrong place for SA.
3. **Atomic-data uncertainty consumption.** The legacy solver folds `aki_uncertainty` into fit-space σ in quadrature
   (`_line_y_uncertainty`, `aki_uncertainty_weighting`). The fast GN **stores** `line_aki_uncertainty` but **never
   uses it** in its objective — the grade data is dead weight on the fast path.

**Operating rule.** All accuracy-changing physics (Levers 1–6) lands in the **legacy iterative path first**, is
benchmark-gated there, and only then is ported to the fast GN under the existing jitpipe parity-test discipline. The
fast GN is not abandoned — it is the eventual latency vehicle (M10) — but it inherits validated physics; it does not
pioneer it. This also matches the physics-only and non-regression constraints: the legacy path is `numpy`/`scipy`, has
the richest `quality_metrics`, and is where defensible CIs matter most.

---

## 4. Per-Lever Program (Ordered by Priority)

Each lever lists: what exists in `cflibs` (verified symbols/files), the gap, the scientifically-grounded change, and
the exact benchmark-gate (flag · datasets · metric · accept criterion). **All gates obey the non-regression rule** (the
repo regressed 3× on ungated scoring changes): a change ships behind its own flag, default-off, and must not regress any
dataset.

---

### P0 — Lever 1: Atomic Data Quality & gf-Grade-Aware Line Selection

**Owns the dominant measured error (0.171 RMSE). This is the single highest-value lever.**

**What exists (verified):**
- `LineSelector` / `LineScore` / `LineSelectionResult` — `cflibs/inversion/physics/line_selection.py`. Grade-aware
  **soft** scoring only: `score = SNR · (1/σ_atomic) · isolation`; `ATOMIC_UNCERTAINTIES` grade→σ map (AAA…E).
  **No grade threshold / hard gate anywhere** (confirmed by grep across `cflibs/`).
- `LineObservation.aki_uncertainty` — `cflibs/inversion/common/data_structures.py` (carries the grade into the solver).
- `IterativeCFLIBSSolver.aki_uncertainty_weighting` / `_line_y_uncertainty` —
  `cflibs/inversion/solve/iterative.py` (the **one path that actually uses** grade data, folds it into fit-space σ).
- `AtomicLine.accuracy_grade` schema + `_backfill_uncertainty` — `cflibs/atomic/structures.py`,
  `cflibs/atomic/database.py`. **PARTIAL / DEAD-data:** schema exists but grades are **heuristic for ~99.7% of lines**
  (rel-int → B/C/D/E backfill); **only 72/28,727 lines carry real NIST-derived unc** (0.03/0.07).
- `datagen_v2.py` NIST scrape — reads `obs_wl_air` + `Aki`; **drops `acc`/`unc_Aki` for nearly all lines** (root cause
  of fabricated grades). Stores **air** wavelengths.
- `line_aki_uncertainty` in the structured-GN — `cflibs/jitpipe/snapshot.py`, `host.py`. **STORED but UNUSED** in the
  GN objective.
- `AtomicDataSource` ABC — `cflibs/atomic/database.py` (the natural seam for a graded Kurucz/VALD3 backend; only
  SQLite/NIST implemented).
- **Air↔vacuum conversion — MISSING ENTIRELY** (verified: no `edlen`/`ciddor`/`vac2air`/`air2vac` anywhere in
  `cflibs/`). Datagen stores `obs_wl_air` with no enforced convention vs the forward model.

**The gap.** The transition metals that dominate geological LIBS (Fe 7312 lines, Ti, V, Cr, Mn, Ni) have **zero real
grade-A lines**. The entire gf-grade-aware machinery is selecting and weighting on **invented grades** — the worst-case
input-quality failure for a calibration-free method.

**The change (three ordered, independently-flagged sub-levers):**

- **(A) Fix data provenance first (root cause).** Make the NIST ASD query in `datagen_v2.py` request and parse the
  `acc` and `unc_Aki` columns per line and persist them verbatim; fall back to the rel-int heuristic **only** when NIST
  truly has no grade. This converts the existing grade-aware scoring and the legacy solver's `aki_uncertainty_weighting`
  from fabricated to real, **at zero solver-math risk** (Ciucci 1999: per-line gf error ~1:1 into intercept).
- **(B) Add a hard grade threshold** as a selection gate (none exists today). Add `max_atomic_uncertainty` / `min_grade`
  to `LineSelector` that — **where ≥`min_lines_per_element` grade-A/B lines with adequate E_k spread exist** — restricts
  the Boltzmann/Saha quantitation fit to them and demotes C+/D/E lines to identification-only. Must be a graceful soft
  gate, never an unconditional cutoff (see risks).
- **(C) Add a curated completeness backend behind a flag.** A Kurucz (and later VALD3) `AtomicDataSource`
  implementation, used **only to fill gaps** where NIST has no graded value, with explicit air↔vacuum normalization.
  **A new Edlén/Ciddor converter MUST be added and applied at ingest** so the line list, wavelength solution, and
  forward model share one convention (Ciucci/Tognoni). Kurucz/VALD lines carry a synthetic grade `U` (unknown) and are
  quantitation-eligible only when no NIST-graded alternative exists.

Wire all of this into the **legacy iterative path first** (it already consumes `aki_uncertainty_weighting`); wire the
fast GN only after the legacy path validates.

**Benchmark-gate.** Three independent flags.
- **Datasets:** synthetic round-trip corpus (controlled ground truth, isolates the atomic-data term toward 0.171) +
  real ChemCam + SuperCam (`validate_real_data.py`).
- **Metric:** composition RMSE (mass-fraction) primary; Aitchison/ILR distance secondary; report T/n_e drift to confirm
  no regression.
- **Accept:** (A) regenerate DB; grade distribution sanity (real NIST grades must produce **>0 grade-A for Fe-group**
  vs current 0) AND synthetic RMSE with real grades ≤ RMSE with heuristic grades. (B) flag `use_grade_threshold`
  (default off); accept iff real-data median RMSE improves on ≥3/5 spectra with no synthetic regression. (C) flag
  `atomic_source=kurucz` vs `nist`; accept on the already-measured target: **SuperCam composition improves ~5–6 wt% vs
  NIST** with T/n_e stable and no synthetic regression. All three must pass `pytest -m physics` + a new narrow
  air/vacuum-roundtrip unit test. Full suite via parent-backgrounded job, never inside a sub-agent.

---

### P1 — Lever 2: Instrument Calibration (ADR-0006) & Matrix Effects

**Second-largest error, currently UNMEASURED in-repo — the first deliverable is to measure it.**

**What exists (verified):**
- `InstrumentModel` / `InstrumentModelJax` (forward E(λ) multiply), `apply_response`, `apply_instrument_function`
  (Gaussian LSF, **single σ only**) — `cflibs/instrument/`.
- `SpectralResponseCorrection` (inversion-side divide by E(λ), coverage/extrapolation guards, uncertainty
  propagation) — `cflibs/inversion/preprocess/response_correction.py`. **Wired into legacy inversion**
  (`pipeline.py:816–820`) and into jitpipe (`host.py`).
- `derive_response_from_argon_branching_ratios` — `response_correction.py:363`. **DEAD stub** (raises
  `NotImplementedError`).
- `calibrate_wavelength_axis` (segmented RANSAC+BIC + CCD seam detection) — `preprocess/wavelength_calibration.py`.
- `MatrixEffectCorrector` / `InternalStandardizer` / `CorrectionFactorDB` — `cflibs/inversion/physics/matrix_effects.py`.
  **DEAD CODE:** exported + one isolated test, **zero pipeline/solve/benchmark callers**, generic uncalibrated `n_s`
  factors.
- Chebyshev baseline + resolving-power nuisance in `bayesian/` — **additive** baseline, **NOT** a multiplicative E(λ)
  fit.
- `InstrumentCalibration` object + builders (ADR-0006) — **MISSING** (verified: `cflibs/instrument/calibration.py` does
  not exist).

**The gap.** No isolation experiment quantifying the E(λ) contribution; **no response-curve data files in `data/`** (so
every real-data benchmark is genuinely Mode-C); the argon branching-ratio derivation is a stub; no measured/variable
LSF (only fixed-FWHM/resolving-power — instrument width can be over-attributed to Stark n_e, the n_e-rails failure);
matrix correction is dead code with uncalibrated factors; no calibration-mode column on the scoreboard.

**The change (measurement-first; do NOT start with the full ADR-0006 object):**

1. **MEASUREMENT FIRST.** Derive an E(λ) — finish `derive_response_from_argon_branching_ratios` (argon
   branching-ratio method, J. Anal. At. Spectrom. 29 (2014) 657-664 + Whaling 1993 tables) **or** inject a known E(λ)
   on the synthetic corpus — and report composition RMSE with `response_curve` OFF vs ON through the **legacy**
   pipeline. This quantifies the lever and *is* the gate.
2. **Provide/derive curves for non-vendor lab datasets**; label vendor-corrected ChemCam/SuperCam as **Mode B** and
   **do not double-correct** (the hook correctly defaults to `None` for CCS — preserve and test that guard).
3. **Add a measured LSF path** (Voigt / σ(λ) fit from isolated lamp/self lines) so width is not over-attributed to
   Stark n_e.
4. **Resolve matrix_effects.py:** wire `InternalStandardizer` as a *gated* post-step for closure-breaking matrices,
   backed by **matrix-matched** factors — or mark it explicitly non-default/experimental. Generic `n_s` factors must
   not ship as a default.
5. **Defer** the full `InstrumentCalibration` object and the Mode-C multiplicative E(λ) self-fit until (1)–(2) prove
   the lever's size. Mode-C smoothing, if built, is explicit Chebyshev/spline/Savitzky-Golay — **never `jax.nn`** — and
   carries a loud non-quantitative flag (ADR-0006 D3).

**Benchmark-gate.**
- **Flag:** `response_curve` (existing pipeline knob) — OFF (`None`) vs ON (derived/measured E(λ)).
- **Datasets:** (a) synthetic corpus with a **known injected E(λ)** (exact RMSE) as the primary gate; (b) in-house lab
  sets lacking vendor correction (`nist_steel`, `silva2022_tropical_soils`, `nist_srm_612`) for confirmation. Solver:
  legacy `run_pipeline`, geological preset, `stark_ne=True`.
- **Metric:** composition RMSE (mass-frac) + Aitchison/ILR; also report fitted-T shift (slope-rotation check).
- **Accept:** flag-ON must reduce RMSE on the synthetic-known-E(λ) set by a pre-registered margin (≥10% relative, no
  regression on any dataset) AND must **not** increase RMSE on vendor-corrected ChemCam/SuperCam (which stay Mode-B,
  response OFF — double-correction guard). Add a **calibration-mode column** to the real-data scoreboard and enforce
  ADR-0006 D3: **Mode-C results are non-quantitative and cannot flip a default.**

---

### P1 — Lever 5: Stark Broadening & Electron-Density Diagnostics

**Headline gain (pin/seed n_e) already BANKED — this lever is incremental coverage + bias-correction. Correctly P1,
not P0.**

**What exists (verified):**
- `measure_stark_ne` — `cflibs/inversion/physics/stark_ne.py`. Literature-grade (`stark_b`) gating, pinned-Gaussian
  Voigt fit, instrument-FWHM ladder, Doppler removal, multiplet-blend DB gate, resonance down-rank, median+1.4826·MAD
  combine. `LITERATURE_STARK_SOURCES = ("stark_b",)` (verified).
- `estimate_ne_from_stark` / `stark_width` / `stark_hwhm` — `cflibs/radiation/stark.py` (single source of truth for the
  width law; A4-CONV-2 20× over-broadening bug documented as fixed).
- `_update_ne_python` / `_estimate_ne_from_stark_multi` — `cflibs/inversion/solve/iterative.py` (legacy solver consumes
  Stark as **primary** per-iteration n_e, 50/50 damped; pressure-balance only as warned fallback).
- `measure_stark_ne_jit` (J6 on-device port) — `cflibs/jitpipe/stark.py` (parity-tested).
- `solve_python` n_e pin — `cflibs/jitpipe/solve.py` (fast GN **pins** n_e = `ne_stark_cm3`).
- `stark_ne=True` in geological/default presets (verified `pipeline.py:63`).

**The gap (binding).** **Only 244/28,727 DB lines (0.85%) carry `stark_b` provenance** (the rest:
`konjevic_lambda_sq_scaled` 22,951, `interpolated` 4,574, `hydrogenic` 562). On many real spectra 0–2 `stark_b` lines
survive the SNR/isolation/resonance gates ⇒ diagnostic returns `usable==False` ⇒ solver silently falls back to the
**physically-invalid 1-atm pressure balance** (the exact F2 failure the lever was built to remove). Secondary: H-α
stored as `stark_b` α=0.2 under the **linear-in-n_e** convention, but Balmer widths genuinely scale **~n_e^0.7**
(Gigosos 2014) — inverting H-α with n_e ∝ w^1.0 biases n_e. Tertiary: no in-solve Stark-vs-Saha n_e self-consistency
flag.

**The change.**
- **PRIMARY:** expand literature-grade Stark-width coverage. Promote `scripts/archive/migrations/ingest_stark_b.py` to
  a maintained data step; ingest critically-evaluated widths (Konjevic 2002 + STARK-B / Sahal-Bréchot) for the
  canonical LIBS diagnostics across the full 240–850 nm SuperCam/ChemCam window, tagged `stark_w_source='stark_b'`.
  Each conversion of a silent pressure-balance fallback into a Stark-pinned solve **is** the measured ~2× legacy-vs-GN
  delta. **Do NOT relax the gate** to admit `konjevic_lambda_sq_scaled` for *measurement* — widen the trusted set by
  improving the data, not the gate.
- **SECONDARY:** implement the H-α n_e^0.7 exponent (per-line Balmer flag in the inverse path, mirrored in the forward
  path to preserve round-trip symmetry).
- **TERTIARY (reliability):** add an in-solve Stark-vs-Saha n_e self-consistency QC flag (set
  `quality_metrics['stark_saha_inconsistent']=1.0` when they diverge by >0.3 dex). **Report-only** — never wire it back
  into the loop (that risks the n_e-rails instability).

**Benchmark-gate.**
- **Flag:** `--stark-coverage {legacy, expanded}` (legacy = current 244-line DB; expanded = newly-ingested set).
  Separate `--halpha-exponent {linear, balmer}`. Stark-Saha flag is report-only.
- **Datasets:** real ChemCam + SuperCam (where legacy scores 0.076 vs GN 0.151) + synthetic round-trip corpus
  (controlled n_e ground truth).
- **Metric:** (1) composition RMSE; (2) fraction of spectra where `ne_from_stark==True` (firing rate — the mechanism);
  (3) synthetic |n_e − n_e_true|/n_e_true (target 10–20%, down from ~30%).
- **Accept:** expanded coverage must (a) strictly increase the firing rate, (b) **not regress composition RMSE on any
  dataset**, and (c) improve median composition RMSE on ≥1 real dataset OR improve synthetic n_e RMSE by ≥5 percentage
  points. H-α exponent: accept iff synthetic n_e RMSE improves with no composition regression. Stark-Saha flag: must
  fire (true-positive) on known-non-LTE spectra and stay silent on clean round-trips, **zero change to T/n_e/composition
  outputs**. Every ingested width must pass `tests/test_stark_provenance.py` published-value anchors (the A4-CONV-2 20×
  history is the cautionary tale).

---

### P1 — Lever 6: LTE Validity, Plasma Homogeneity & Multi-Zone Reliability

**Small share of current RMSE; owns the reliability/trust surface, which is presently un-enforced. Cheapest to wire —
the physics code already exists.**

**What exists (verified):**
- `LTEValidator` (`check_mcwhirter`, `check_temporal`, `validate`) — `cflibs/plasma/lte_validator.py`. McWhirter wired
  into the solver; **temporal defaults OFF, never enabled in production**; δE uses `max(E_k)` not the resonance gap
  (under-estimates required n_e ⇒ lets non-LTE plasmas pass).
- `IterativeCFLIBSSolver._assemble_quality_metrics` LTE block (`iterative.py:2005–2044`) — merges
  `lte_mcwhirter_satisfied` / `lte_n_e_ratio` into the result; **necessary condition only**.
- `QualityAssessor.assess` / `QualityMetrics` / `_determine_quality_flag`
  (excellent/good/acceptable/poor/reject) — `cflibs/inversion/physics/quality.py`. **DEAD IN PRODUCTION:** the full
  Cristoforetti necessary-not-sufficient gate (Saha-Boltzmann consistency, inter-element T scatter, closure thresholds)
  has **zero call sites** outside tests.
- `physical_consistency` (McWhirter floor + multi-T LTE consistency) — `cflibs/benchmark/physical_consistency.py`.
  Wired as a corpus-level benchmark gate; but `check_lte_consistency` is effectively always N/A (solver emits a single
  T).
- `TwoZoneMCMCSampler` — `cflibs/inversion/solve/bayesian/two_zone.py`. Exists, **unwired**, self-absorption-scoped,
  not an LTE/gradient fallback.
- CLI quality surfacing (`cli/main.py:282–296`) — McWhirter failure is a **soft warning only**; the "RESULT UNRELIABLE"
  decision (line 294) keys **only** on converged/boltzmann_degenerate/closure_degenerate — **LTE failure does not mark
  a result unreliable.**

**The gap.** No refuse-to-report default; the Cristoforetti multi-check is dead; the solver emits a single excitation
temperature (multi-T LTE consistency permanently N/A); McWhirter δE under-estimates the floor; no automatic multi-zone
fallback.

**The change (two phases).**
- **PHASE 1 (P1, cheap — wire existing dead code into a refuse-to-report default):** (a) call `QualityAssessor.assess`
  inside `_assemble_quality_metrics` and merge `quality_flag` + `saha_boltzmann_consistency` +
  `inter_element_t_std_frac`; (b) fix the McWhirter δE to use the true resonance ground→first-excited gap (from
  `energy_levels`) instead of `max(E_k)`; (c) add an `overall_reliable` boolean to `CFLIBSResult` ({McWhirter satisfied
  AND quality_flag ∈ (excellent, good, acceptable)}) and add it to the CLI "RESULT UNRELIABLE" gate so the CLI refuses
  to present a confident composition below acceptable; (d) optionally enable `check_temporal` when a gate-delay/lifetime
  is known.
- **PHASE 2 (P3, prove-before-trust):** wire `TwoZoneMCMCSampler` as an **opt-in** fallback triggered only when Phase-1
  flags inhomogeneity, and benchmark whether two-zone beats biased single-zone on the flagged subset. **Do NOT make
  two-zone a default** (it adds identifiability problems — n_e is already unidentifiable from a flat prior).

Host all Phase-1 gates in the legacy iterative solver (the 2×-more-accurate path with the richest `quality_metrics`).

**Benchmark-gate.**
- **Flag:** `CFLIBS_REFUSE_TO_REPORT` (default OFF for the gate run). Each sub-change (assess-wiring, δE fix, CLI gate)
  ships behind its own toggle and is benched independently.
- **Datasets:** real ChemCam + SuperCam (the 5 spectra) + synthetic corpus (`output/synthetic_corpus`, ground-truth
  T/n_e to label known-LTE vs known-non-LTE).
- **Metric:** (i) fraction of spectra at each quality_flag; (ii) composition RMSE flag-OFF (report-everything) vs
  flag-ON (conditioned on quality_flag ≥ acceptable, refused excluded); (iii) synthetic confusion — does the gate flag
  spectra whose injected n_e is below the resonance-gap McWhirter floor?
- **Accept:** flag-ON conditional RMSE on the passing subset ≤ flag-OFF unconditional RMSE; **zero false-reject on
  known-good synthetic LTE spectra**; gate **must** flag the SuperCam 20–26 wt% outliers if they trip ≥2
  physical-consistency checks. The δE fix is separately gated: re-run the corpus and confirm the n_e-required floor
  rises (more spectra correctly flagged) **without changing any composition value** (the fix touches only the flag).

---

### P2 — Lever 4: Self-Absorption

**Moderate, conditional, and never benchmark-gated in-repo. Do NOT add new physics first — PROVE the existing
observable corrector, then wire it into the preset only if it wins.**

**What exists (verified):**
- `ObservableSelfAbsorptionCorrector` — `cflibs/inversion/physics/self_absorption_observable.py`
  (**production path**; the composition-fed corrector was deliberately deleted per bead 0jvr).
- Planck-ceiling path (`correct_intensity_planck`, Voelker & Gornushkin 2023) — present but **inert on uncalibrated
  data** (needs absolute peak radiance).
- `correct_via_doublet_ratio` / `find_doublet_pairs` (Pace 2025) — `self_absorption.py`.
- `CurveOfGrowthAnalyzer` / `estimate_optical_depth_from_intensity_ratio` — `self_absorption.py`
  (**diagnostic-only**; composition-fed `.correct()` deleted).
- Legacy solver SA wiring (`iterative.py:1573–1576`) — corrector applied **once before** the Boltzmann fit
  (correct placement: upstream, observable-gated, **not** inside the fast GN).
- `--apply-self-absorption` CLI flag (`cli/main.py:968`).
- **`ANALYSIS_PRESETS['geological']` has SA OFF** (verified: `apply_self_absorption` default-off in every preset incl.
  the measured-best geological).
- `score_spectrum(config_overrides=...)` / `run_scoreboard` — gate harness ready.

**The gap.** **Benchmark gate never run** — zero SA references in `cflibs/benchmark/`. A wired, audited,
deleted-feedback-loop-replacement corrector that production never invokes. No C-σ graph (Aguilera-Aragon 2014) / CD-SB
joint solver. No El Sherbini 2005 Stark-width-ratio τ path (the highest-yield missing observable, since doublet pairs
are rare on real detected lists). No IRSAC internal-reference path. The scoreboard does not report whether SA actually
fired (null-effect vs no-pairs indistinguishable).

**The change (sequence — prove, then wire, then add the one high-yield observable):**
1. Add an `apply_self_absorption` override to `run_scoreboard` (`config_overrides={'apply_self_absorption':'observable'}`
   reaches `build_pipeline_config`); run the existing real ChemCam/SuperCam datasets flag-off vs flag-on, **reporting
   n_corrected/n_suspect/max_tau per spectrum** so null-effect is visible.
2. **If and only if** flag-on improves the optically-thick spectra (esp. the Al 877 case) **without regressing the thin
   majority**, add `'apply_self_absorption':'observable'` to `ANALYSIS_PRESETS['geological']` and `'metallic'`.
3. **Then** add the El Sherbini 2005 Stark-width-ratio τ path inside `ObservableSelfAbsorptionCorrector` (the DB already
   exposes Stark widths via `stark_ne`) — it supplies per-line τ **without** needing a doublet pair. Keep it
   observable-gated (τ from measured/Stark width ratio, **never** from composition); mask τ>3.
4. **DEFER** the C-σ / CD-SB joint solver to a separate bead (large new physics surface; the directive says perfect the
   legacy path first). Keep **all** SA upstream in the legacy path and **out of the fast GN** (modeling SA inside the GN
   rails n_e).

**Benchmark-gate.**
- **Flag:** `apply_self_absorption` (`off` baseline vs `observable`), threaded
  `run_scoreboard → score_spectrum(config_overrides) → build_pipeline_config`.
- **Datasets:** real ChemCam + SuperCam (geological preset, `stark_ne=True`, `pipeline_impl='reference'` — SA lives in
  the legacy path, **not** jit) **plus** a synthetic optically-thick positive control (attenuate a known doublet by
  physical escape factors, per `test_self_absorption_wiring.py`).
- **Metric:** median per-spectrum composition RMSE (wt%); secondary per-spectrum n_corrected/n_suspect/max_tau and
  signed per-element error on the optically-thick majors (Ca/Na/Al/Fe).
- **Accept (ALL required):** (a) synthetic positive control RMSE strictly decreases flag-on; (b) median real-data RMSE
  does **not** increase (non-regression on the thin majority is mandatory — 3 prior regressions); (c) on SA-heavy
  spectra where n_corrected>0 (e.g. Al 877), RMSE decreases or the resonance-intercept bias shrinks; (d) mode `off`
  remains bit-identical to default. Only flip the preset default after (a)–(d) pass.

---

### P2 — Lever 6/7: Uncertainty Calibration & Precision Reporting

**~0% of the point-estimate gap; ~100% of the trustworthiness gap. The engine is strong — close the three reporting
gaps. Strictly a reliability lever; never sold as an RMSE reducer.**

**What exists (verified):**
- `MonteCarloUQ` / `run_monte_carlo_uq` / `MonteCarloResult` — `cflibs/inversion/physics/uncertainty.py`. Full-pipeline
  re-run with SPECTRAL_NOISE/ATOMIC_DATA/**COMBINED** perturbation (COMBINED is default, includes A_ki); Gaussian +
  Poisson noise; T/n_e/conc means+std+CI_68+CI_95; `correlation_matrix`; `compare_with_bayesian`.
- Analytical path: `create_boltzmann_uncertainties` / `propagate_through_closure_*` (correlation-aware via the
  `uncertainties` package; correlated_values from the 2×2 Boltzmann covariance).
- `IterativeCFLIBSSolver.solve_with_uncertainty` — analytical UQ for the legacy solver. **CIs only populated by
  `solve_with_uncertainty`**; plain `solve()` / fallbacks set them to 0.0/{}/None.
- `AtomicDataUncertainty` (NIST grades → fractional A_ki uncertainty) — used by MC ATOMIC_DATA/COMBINED.
- `PosteriorDiagnostics` (coverage_95, coverage_in_band band **[0.93, 0.97]** — verified, PIT, sharpness, PSIS-LOO,
  R-hat, ESS) — `cflibs/benchmark/posterior_metrics.py`. **BAYESIAN-ONLY** (invoked via
  `_maybe_compute_posterior_diagnostics`). **Not applied to classical/MC CIs.**
- `QualityMetrics._determine_quality_flag` — derives the flag **only** from
  r_squared/saha_consistency/t_std_frac/closure_residual. **Completely decoupled from uncertainty/CI width** ⇒ a weak
  emitter with a huge CI can be flagged "good".

**The gap.** No coverage/calibration check for the **classical** path — analytical and MC CIs are never validated
against synthetic-corpus truth; given the ~0.20 weak-emitter floor, the analytical Gaussian CIs almost certainly
**under-cover** for Na/K/Si (overconfident-and-wrong — the worst failure mode). The budget is incomplete: A_ki enters
analytically only as a per-line weight (no correlated σ_gf systematic), and self-absorption uncertainty propagates in
neither path.

**The change (do NOT add new estimators):**
1. **Couple uncertainty to the reliability flag:** extend `_determine_quality_flag` with an uncertainty-aware tier that
   **downgrades** quality_flag to poor/reject when an element's relative concentration uncertainty exceeds a threshold
   (>50% → poor, >100% → reject). Add per-element reliability flags to `CFLIBSResult`. (Thresholds tuned on the
   synthetic corpus, not guessed — gate on the confusion metric.)
2. **Extend the coverage gate to the classical path:** reuse `posterior_metrics._coverage_per_param` /
   `_coverage_verdict` (band [0.93, 0.97]) to compute empirical 95%-CI coverage of `solve_with_uncertainty` +
   `MonteCarloUQ` outputs against synthetic-corpus truth.
3. **Complete the budget:** make the COMBINED/atomic-data term the documented default for the analytical path (add a
   correlated σ_gf systematic per element via `AtomicDataUncertainty.from_transitions`), and propagate a
   self-absorption uncertainty term **by perturbing the SA correction factor in `MonteCarloUQ`** (outer-loop re-runs of
   the legacy solver) — **never inside the fast GN** (rails n_e).

Prefer the legacy solver's `solve_with_uncertainty` (2× more accurate; defensible CIs matter most there).

**Benchmark-gate.**
- **Flag:** `--uq-coverage-gate` (off) + `reliability_from_uncertainty`, `analytical_atomic_uncertainty`,
  `sa_uncertainty_mc` (all default off).
- **Datasets:** synthetic ID/composition corpus (known truth ⇒ coverage computable) + controlled round-trip golden
  spectra (`cflibs/validation` `GoldenSpectrum`).
- **Metric:** (a) empirical 95%-CI coverage_95 of classical CIs vs truth; (b) sharpness (mean CI width, CLR space);
  (c) weak-emitter reliability-flag confusion (does Na/K/Si get downgraded when its CI misses truth?); (d) composition
  RMSE on the scoreboard.
- **Accept:** flag-off == flag-on RMSE **bit-for-bit on the point estimate** (proves no accuracy regression); flag-on
  coverage_95 moves toward/into [0.93, 0.97]; weak emitters that miss truth now carry CIs that bracket truth OR a
  downgraded reliability flag (no silent overconfidence). Watch for the σ_gf systematic pushing coverage to
  *over*-cover (>0.97) — the bidirectional band catches this; balance σ_gf against the existing per-line weighting to
  avoid double-counting A_ki.

---

## 5. Per-Matrix SOTA Accuracy Targets

Literature-best CF-LIBS accuracy by matrix class, and the target this program should hit. Targets are set at or
slightly above the literature SOTA for well-calibrated, optically-thin, LTE conditions; **honest caveat:** these are
reachable only **after** Levers 1–2 land (atomic data + calibration are the binding inputs). Today's measured numbers
are listed for contrast.

| Matrix class | Literature-best CF-LIBS accuracy | Citation | This program's target | Today (v4 measured) |
|--------------|----------------------------------|----------|-----------------------|---------------------|
| **Alloys / steel** (optically-thin, well-characterized) | **~1–3 wt% relative for majors** under well-calibrated, optically-thin, LTE conditions | Tognoni 2010 (10.1016/J.SAB.2009.11.006); Aguilera & Aragón C-σ for optically-thick steel (10.1016/J.JQSRT.2014.07.026); Hou 2019 blackbody-ref on stainless (10.1016/j.aca.2019.01.016, ~27%→~2% class) | **≤3 wt% majors** (Mode-A/B calibrated, grade-A/B line list); SA-corrected for thick lines | In-house lab steel is Mode-C uncalibrated; E(λ) fully uncontrolled (calibration_matrix lever) |
| **Geology / soils / SRM glass** (Fe-group-rich, matrix effects) | **~1–3 wt% norm** for majors when matrix-matched / internally standardized; **10s of % shifts** when matrix effects uncorrected | Tognoni 2010; Hahn & Omenetto 2012 (10.1366/11-06574, matrix effects dominant limiter); Cremers & Radziemski 2013 (10.1002/9781118567371) | **≤5 wt% majors** initially (gated on Lever 1 Fe-group grade-A coverage via Wood 2014 V II / Lawler 2013 Ti I), tightening toward **3 wt%** with matrix-matched internal standardization | ChemCam **0.12–0.17 mass-frac**; legacy median **0.076** (the 2×-better path) |
| **Planetary** (ChemCam/SuperCam, Mode-B vendor radiance-corrected) | Calibration-library / matrix-aware quantitation; instrument response handled upstream | Wan et al. 2025 (10.3847/1538-4357/ae14ef, Zhurong Mars-rover library); Tognoni 2010 | **<10 wt% near-term, target ≤5 wt%** for line-rich SuperCam after Kurucz completeness (measured −5/−6 wt% gain) + Stark coverage; residual is atomic-data + weak-emitter + matrix-ablation (E(λ) is Mode-B-handled) | SuperCam **20–26 wt%** (Mode-C uncalibrated, 4–10× over norm); **Kurucz −5/−6 wt% vs NIST** measured |

**Caveat on weak emitters (all matrices).** Na/K/Si at ~10⁶ dynamic range floor at **RMSE ≈0.20** by ill-posedness
(v4 measured). For these the target is **not** a tight RMSE but **calibrated coverage** (Lever 6/7): the reported CI
must bracket truth ~95% of the time, and the per-element reliability flag must downgrade when ill-posed. Tognoni 2007
establishes this floor is intrinsic to CF-LIBS precision even for an ideal plasma.

---

## 6. Reliability & Precision — Making Results "Absolutely Reliable"

Accuracy of the mean is necessary but not sufficient; a scientific result is reliable only when it is **honest about
its own uncertainty** and **refuses to report** when its assumptions are violated. This combines Levers 6 and 7 into one
discipline.

**(A) Refuse-to-report gates (LTE validity).**
- **McWhirter as a necessary condition** with the corrected δE (true resonance ground→first-excited gap, not
  `max(E_k)`), surfaced in `quality_metrics` (Cristoforetti 2010: necessary, **not sufficient**).
- **Cristoforetti multi-check** (currently dead `QualityAssessor.assess`) wired into the production solver:
  Saha-Boltzmann T consistency, inter-element T scatter, closure residual → `quality_flag`
  ∈ {excellent, good, acceptable, poor, reject}.
- **`overall_reliable`** boolean on `CFLIBSResult` ({McWhirter satisfied AND quality_flag ≥ acceptable}), wired into the
  CLI "RESULT UNRELIABLE" gate. Below `acceptable`, **the pipeline refuses to present a confident composition.**
- Quality-flag thresholds (repo `quality.py`, Cristoforetti-grounded): Boltzmann R² >0.95 excellent / >0.80 acceptable;
  Saha-Boltzmann consistency <0.10 / <0.30; inter-element T std/mean <0.05 / <0.15; closure residual <0.01 / <0.10.

**(B) Calibrated per-element uncertainty.**
- Report **both** analytical (correlation-aware via the `uncertainties` package — essential because Boltzmann slope (T)
  and intercept (composition) are correlated from the same regression) **and** Monte-Carlo (`MonteCarloUQ`,
  perturbation-by-noise, Anderson 2025 methodology) uncertainties.
- **Coverage validation** of the classical path against synthetic truth, gated into the repo's own **[0.93, 0.97]**
  band (the protocol band already used for Bayesian).
- **Full budget** (Tognoni 2007 / Safi 2019): spectral-noise + correlated σ_gf atomic-data + T + n_e + self-absorption
  terms. Weak-emitter CIs must **widen** to reflect the ~0.20 ill-posedness floor — large honest CIs, never tight wrong
  ones.

**(C) Couple uncertainty to the reliability flag.** The decisive reliability rule: a result is labeled trustworthy only
when **BOTH** the uncertainty is bounded AND the quality_flag ≥ acceptable. A weak emitter with a huge CI is downgraded
to poor/reject even if its fit metrics look fine. Distinguish **precision** (MC spread, repeatability) from **accuracy**
(bias vs reference) in reporting.

**(D) Stark-Saha self-consistency flag.** Report-only `stark_saha_inconsistent` when Stark-measured n_e and
Saha/charge-balance n_e diverge by >0.3 dex (Cristoforetti 2010 self-consistency). Never wired back into the loop.

**(E) Multi-zone fallback (deferred, opt-in).** When Phase-1 flags inhomogeneity, optionally route to
`TwoZoneMCMCSampler` and benchmark whether two-zone beats biased single-zone on the flagged subset. Prove before
trusting; never default.

---

## 7. Sequenced Milestones (M1…M10)

Accuracy milestones first; latency last and explicitly gated. Each milestone is benchmark-gated per §4 and obeys the
non-regression rule.

| Milestone | Lever(s) | Deliverable | Gate criterion |
|-----------|----------|-------------|----------------|
| **M1 — Real atomic-data provenance** | 1A | Capture NIST `acc`/`unc_Aki` in `datagen_v2.py`; regenerate DB; real grades replace heuristic for ~99.7% of lines | **>0 grade-A for Fe-group** (vs current 0); synthetic RMSE with real grades ≤ heuristic; T/n_e stable |
| **M2 — Grade threshold + air/vacuum** | 1B, 1C | Hard `min_grade` soft-gate in `LineSelector`; Edlén/Ciddor converter at ingest; one-convention enforcement | Real-data median RMSE improves ≥3/5 spectra, no synthetic regression; air/vacuum-roundtrip unit test passes |
| **M3 — Calibration measurement + E(λ)** | 2 | Finish argon branching-ratio E(λ) derivation OR synthetic known-E(λ); calibration-mode scoreboard column; Mode-B double-correction guard | Flag-ON reduces synthetic-known-E(λ) RMSE ≥10% relative, no regression; ChemCam/SuperCam stay Mode-B/response-OFF |
| **M4 — Stark coverage + H-α exponent** | 5 | Ingest critically-evaluated `stark_b` widths across 240–850 nm; H-α n_e^0.7 exponent; Stark-Saha report-only flag | Firing rate ↑, no composition regression on any dataset, (≥1 real RMSE ↓ OR synthetic n_e RMSE ↓ ≥5 pts); provenance anchors pass |
| **M5 — Kurucz completeness backend** | 1C | `AtomicDataSource` Kurucz backend behind a flag (fill NIST gaps; grade `U`; vacuum-correct) | **SuperCam ~5–6 wt% improvement** (the measured target), T/n_e stable, no synthetic regression |
| **M6 — Self-absorption gated + wired** | 4 | Score observable SA flag-off vs flag-on; El Sherbini width-ratio τ; wire into geological/metallic presets if it wins | Synthetic positive-control RMSE ↓; real-data non-regression; SA-heavy (Al 877) RMSE ↓ or intercept-bias ↓; mode-off bit-identical |
| **M7 — LTE refuse-to-report (Phase 1)** | 6 | Wire `QualityAssessor.assess`; fix McWhirter δE; `overall_reliable` + CLI gate | Conditional RMSE ≤ unconditional; zero false-reject on known-LTE synthetic; flags SuperCam 20–26 wt% outliers |
| **M8 — Classical-path coverage + reliability coupling** | 7 | Extend coverage gate to classical/MC CIs; couple quality_flag to CI width; per-element reliability flags | RMSE bit-identical flag-on/off; coverage_95 → [0.93, 0.97]; weak emitters bracket truth or downgrade |
| **M9 — Full uncertainty budget** | 7 | Correlated σ_gf systematic (analytical) + SA-uncertainty MC term | Coverage stays in-band (not over-covering); no point-estimate change; budget completeness documented |
| **M10 — LATENCY (DEFERRED, gated)** | fast GN | Port validated physics (Stark pin, grade weighting, observable-SA upstream) into the structured-GN under parity gates; pursue sub-ms | **Gated on M1–M9 complete and locked.** Fast-GN composition RMSE must **match the legacy path within tolerance** on real data (close the measured 0.151 vs 0.076 gap) before any latency claim. Latency improvements must **not** regress accuracy — parity-tested, scoreboard-gated. |

**Phase-2 (post-M9, prove-before-trust, not on the critical path):** wire `TwoZoneMCMCSampler` as an opt-in
inhomogeneity fallback (Lever 6 Phase 2); add a C-σ / CD-SB joint optically-thick solver (Lever 4 deferred); add a
real-replicate repeatability check (Lever 7 caveat). Each is a separate benchmark-gated bead.

**Explicit deferral statement.** M10 is the only latency milestone and it is **last**. It cannot start until M1–M9 have
landed and the accuracy scoreboard is locked. The fast structured-GN is presently 2× less accurate on real data;
promoting it before the accuracy program is complete would violate the standing directive (trading accuracy for speed).
Sub-ms is pursued **only** once the fast path matches the legacy path's accuracy.

---

## 8. Open Blockers

1. **VALD3 (human-gated).** VALD3 is untested in v4 and gated; it requires registered access and a human decision. Ship
   it **flag-only, never default**, behind the same `AtomicDataSource` seam as Kurucz, with the same air↔vacuum
   normalization and synthetic `U` grade. **Blocker:** access + human sign-off; do not ship as default. (atomic_data
   lever)

2. **Response-curve data / E(λ) derivation.** No response-curve files exist in `data/`, and
   `derive_response_from_argon_branching_ratios` is a dead stub. The calibration lever (Lever 2) **cannot be measured**
   until either the argon branching-ratio derivation is implemented or known-E(λ) is injected on the synthetic corpus.
   **Decision needed:** which lab datasets get derived curves; tungsten-halogen lamp validation availability.
   (calibration_matrix lever)

3. **STARK-B / Konjevic 2002 width ingestion.** `ingest_stark_b.py` is archived
   (`scripts/archive/migrations/`) and incomplete; only 0.85% of DB lines carry `stark_b` provenance. **Blocker:**
   promote and complete the ingest with published-value anchor tests (the A4-CONV-2 20× over-broadening history is the
   cautionary tale). **Data dependency.** (stark_ne lever)

4. **Fe-group laboratory log(gf).** Fe/Ti/V/Cr/Mn/Ni have zero real grade-A lines. Integrating Wisconsin/Lund
   FTS+lifetime campaigns (Wood 2014 V II ~few-%; Lawler 2013 Ti I) is the path to grade-A Fe-group coverage but is a
   **data-curation dependency**, not a code change. (atomic_data lever)

5. **DB regeneration coordination.** Regenerating the DB (M1) is hours-long and changes a `frozen_manifest` sha used for
   campaign pinning. **Coordinate** so in-flight benchmark runs are not invalidated mid-run. (atomic_data lever risks)

6. **Single-temperature solver limits multi-T LTE consistency.** The solver emits one excitation temperature, so the
   strongest homogeneity signal (neutral-stage vs ion-stage T) is permanently N/A in the benchmark gate. A
   stage-resolved temperature output is a **design decision** required before the multi-T LTE check can fire.
   (lte_reliability lever)

7. **Reliability-downgrade thresholds need empirical tuning.** The >50%/>100% relative-uncertainty thresholds (Lever 7)
   are heuristic and must be tuned on the synthetic corpus coverage data, not guessed; an over-strict gate over-flags
   precise weak-emitter results. **Tuning dependency on M8 coverage data.** (uncertainty_precision lever)

8. **Mode-C non-quantitative enforcement.** ADR-0006 D3 requires that Mode-C (uncalibrated) results cannot flip a
   default or claim a SOTA number. This must be enforced on the scoreboard as a hard column-level rule — the discipline
   that prevents an uncalibrated regression from masquerading as an algorithm result (the repo's 3×-regression failure
   class). **Decision:** enforce as a CI gate, not just a label. (calibration_matrix lever)

---

### Cross-cutting invariants (apply to every lever)

- **Non-regression is mandatory** (3 prior ungated regressions): every change ships behind its own flag, default-off,
  and must not regress any dataset on the scoreboard.
- **Physics-only:** all proposed work is pure `numpy`/`scipy` + the `uncertainties` package + data curation. No banned
  ML imports. Mode-C smoothing (if ever built) is explicit Chebyshev/spline, never `jax.nn`.
- **Legacy-first:** accuracy physics lands in the legacy iterative path, is gated there, then ported to the fast GN
  under parity tests.
- **Full suite via parent-backgrounded job, never inside a sub-agent** (stream-idle watchdog); sub-agent gates use
  narrow `pytest -m physics` subsets.
- **Honest uncertainty about the program itself:** no single literature value quantifies the exact composition-RMSE
  drop for *this* pipeline from a curated graded list, from E(λ) correction, or from expanded Stark coverage. Each
  lever's headline number is therefore an **isolation experiment to be run** (M1/M3/M4), bounded above by the measured
  0.171 atomic-data term, not an asserted result.
