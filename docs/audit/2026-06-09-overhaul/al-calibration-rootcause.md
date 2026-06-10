# Root cause: Al over-attribution under wavelength calibration vs Mg deletion without it

**Date:** 2026-06-10. **Branch:** `overhaul/wave2-integration` (worktree `.worktrees/w2-integration`).
**Spectrum:** `data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv`. **DB:** `ASD_da/libs_production.db`.
**Config:** geological preset (oxide closure + SB-graph + stark-n_e), confounders requested.
All runs reproduced through `scripts/measure_bhvo2_presence.py` (provenance line confirmed:
`cflibs=.../w2-integration/cflibs`) and `/tmp` instrumentation of
`cflibs.inversion.pipeline.detect_and_select_lines` (monkey-patched
`_register_observation`, `_shift_coherence_veto`, `_select_accepted_elements`).

## Reproduction (this worktree, stark-n_e default-on)

| run | T (K) | n_e | Al wt% (n) | Mg wt% (n) | Mn (n) | RMSE |
|---|---|---|---|---|---|---|
| `rc-al-calon` (defaults) | 11529 | 1.18e18 | **19.44 (6)** | 1.70 (9) | 0.44 (5) | 4.499 |
| `rc-al-caloff` | 11074 | 6.44e17 | **6.88 (5)** | **0 (0)** | 0 (0) | 2.843 |
| `rc-al-calon-nostark` | 9783 | 2.0e17 | 21.02 (6) | 1.85 (9) | 0.26 (5) | 4.990 |
| `rc-al-caloff-nostark` | 9871 | 1.9e17 | 7.17 (5) | 0 (0) | 0 (0) | 3.214 |

(The nostark rows are bit-identical to harness-bridge rows A and C.)

**Stark/Saha interaction (isolated):** the Stark n_e (1.2e18 vs assumed 2e17) *shrinks* Al
19.44 vs 21.02 — higher n_e suppresses the Al II fraction in the Saha (1+S) factor
(χ_Al = 5.99 eV), so the +12 wt% inflation is **entirely a line-set effect**, not an
n_e/Saha effect. (n_e also bears no responsibility for the Mg flip: Mg dies cal-off with
stark on or off.)

## The mechanism (falsifiable statement)

Three coupled defects, all in the axis-alignment chain, none in the solver:

1. **VNIR affine extrapolation (calibration defect).** The segmented calibrator finds 3
   ChemCam channels (UV 240.8–340.8, VIO 382.1–469.1, VNIR 473.2–905.6). UV/VIO get
   `shift` fits (−0.060 / −0.115 nm; rmse 0.021/0.025, 60/35 inliers). VNIR gets an
   **affine** fit (a−1 ≈ −4.6e-4) whose 35–37 inlier pairs sit mostly at 475–650 nm
   (41/56 candidate-pair inliers < 700 nm; the few red "inliers" at 854/877.6/892.6 are
   the disputed peaks themselves, matched circularly within the 2-nm pair window). The
   slope extrapolates the correction from −0.12 @518 nm to **−0.28…−0.34 @877–905 nm**.
   Measured truth at the red end (Ca II IR triplet 849.8/854.2/866.2, the only reliable
   anchors there; VNIR pixel = 0.21 nm): raw−true ≈ +0.06±0.10 — the affine overcorrects
   the red end by ~0.2 nm. Effects: Ca II 854.209 is lost cal-on (present cal-off); the
   Al I 877 doublet flips to the wrong member (below).

2. **Post-calibration residual comb shift-scan runs to its cap (detection defect).**
   After a quality-passed calibration the global comb scan is reduced to ±0.05 nm
   ("mop-up"). Its objective (`_best_summary_improves`: max total F1, then max match
   count; small-|shift| only on exact ties) rewards scooping extra dense-catalog
   coincidences, so it **hits the window edge in every configuration tested**
   (+0.05 default; +0.05 with quadratic-free models; −0.05 with shift-only segments).
   On loc1 the calibrated axis needs ~0 (veto consensus at shift 0 = +0.005 nm); the
   applied +0.05 displaces every element's residual cluster to ≈ +0.044 and, with the
   unchanged ±0.1 tolerance, admits matches up to db−peak = +0.15 — exactly where the
   contaminated Al matches live.

3. **Per-element (not per-line) shift-coherence veto + single global shift (cal-off
   defect).** Without calibration, one global comb shift (−0.10) cannot represent the
   per-channel offsets (UV −0.06 / VIO −0.115 / VNIR −0.12…). Real elements whose lines
   straddle channels become internally bimodal in residual space and fail the ≥50 %
   coherence fraction: **Mg residuals split 11/7 across two clusters → frac 0.44 → VETO;
   Mn frac 0.38 → VETO** (measured veto dump, consensus −0.008, band ±0.033). Al survives
   (frac 0.56) with the *physically right* line set → 6.88 wt% ≈ cert. So cal-off "exact
   Al" and "deleted Mg" are the same coin: the veto judges per-element coherence against
   a single global shift that no piecewise axis can satisfy.

## Per-line evidence: the 6 cal-on Al observations

y = ln(Iλ/gA); Boltzmann expectation from the two resonance anchors with slope −1/kT at
T≈11.5 kK: y_exp(E) ≈ 16.5 − E.

| Al line (cal-on) | E_k (eV) | g | A_ki | I | y | y−y_exp | verdict |
|---|---|---|---|---|---|---|---|
| 394.401 | 3.143 | 2 | 4.99e7 | 1.76e11 | 13.45 | ref | real (resonance) |
| 396.152 | 3.143 | 2 | 9.85e7 | 2.75e11 | 13.22 | ref | real (resonance) |
| 308.215 | 4.021 | 4 | 5.87e7 | 7.95e10 | 11.56 | −0.9 | real (resonance) |
| **256.798** | 4.827 | 4 | 1.92e7 | 5.69e10 | 12.16 | +0.5 | **contaminated: peak true λ 256.696 = Fe II 256.691 (Δ=5 mÅ, Fe 8.6 wt%); Al line is 0.102 nm from the peak — match exists only via the +0.05 shift** |
| **877.287** | 5.434 | 6 | 6.47e6 | 9.01e10 | 14.53 | +3.4 | **wrong doublet member: unresolved Al I 877.287(g=6)+877.390(g=8) blend; −0.34 nm overcorrection puts corrected peak at 877.223, so 877.390 (the stronger member, matched cal-off) is unreachable (Δ=0.117 > tol after shift)** |
| **892.356** | 5.476 | 6 | **2.73e5** | 6.08e10 | **17.32** | **+6.3** | **not Al: gA implies I(892)/I(877) ≈ 1/57 if both Al; observed ratio 1/1.5 (38× too bright). db−peak = +0.105 — match exists only via the +0.05 shift. True emitter ambiguous (raw peak 892.600; nearest credible: K I 892.544 / Co I 892.624 / red-end ripple); no strong line of any sample element is closer than 0.2 nm** |

Cal-off Al set (5): 396.152, 394.401, 308.215, **309.271** (genuine Al I, E 4.02 — lost
cal-on because the +0.05 shift re-routes its peak to Al I 309.284 which then collides with
Mg I 309.298 at the isolation gate), 877.390 (right member). No 256.8/892.4 contaminants
(both are >tol at shift −0.10 on the raw axis).

## Leverage (leave-one-out through the real solver: SB-graph + oxide + stark)

| variant (cal-on obs set) | Al wt% | Mg | RMSE |
|---|---|---|---|
| full (baseline) | 19.43 | 1.70 | 4.497 |
| **− 892.356** | **9.04** | **2.08** | **2.640** |
| − 256.798 | 22.50 (worse: this line's y sits *below* the fit and was pulling Al down) | 1.59 | 5.421 |
| − 877.287 | 14.54 | 1.88 | 3.252 |
| − 892 − 256.8 | 9.27 | 2.07 | 2.639 |
| − 892 − 256.8 − 877 | 3.88 | 2.26 | 3.254 |
| swap 877.287→877.390 (g=8, A=6.95e6) | 18.60 | 1.73 | 4.263 |
| − 892 − 256.8 + swap 877 | 8.56 | 2.09 | 2.649 |

**Al I 892.356 alone carries −10.4 wt% of the over-attribution** (19.43→9.04) and beats
the cal-off RMSE while keeping Mg. The 877 blend carries most of the remaining ~2 wt%
(its +3.x ln excess is intrinsic to attributing an unresolved doublet + possibly
non-Al flux to one member; it affects cal-off too). 256.798 is contaminated but its
leverage is *downward*; removing it alone hurts.

## Fix probes (measured end-to-end on loc1)

| probe | shift | Al (lines) | Mg | Mn | RMSE | notes |
|---|---|---|---|---|---|---|
| A status quo (cal + mop-up 0.05) | +0.050 | 19.43 (6) | 1.70 | 0.44 | 4.497 | |
| B cal + **mop-up = 0** | 0.000 | 10.80 (5) | 2.71 | 0 | **2.353** | drops 892/256.8; veto consensus +0.005; Mg frac 0.83 (15/18). Side effects: **Sn FP 2.45 wt% (6 lines)** — the +0.05 displacement had been *accidentally* decohering Sn; Al steals vetoed Mn's 257.61 peak as Al I 257.539; Mn 5-obs survival in A was itself shift-scoop luck (comb-gate fail at shift 0; A over-attributes Mn 3×) |
| E **shift-only segments** + mop-up 0 | 0.000 | 10.20 (5) | 2.71 | 0 | **2.253** | best probe; picks the correct 877.390 member |
| D shift-only segments, mop-up kept | −0.050 | 22.70 (5) | 2.64 | 0 | 5.360 | scan runs to the *other* edge — the mop-up, not the segment model, is the chaos agent |
| C/F tolerance 0.06 post-cal | 0.000 | 0 (0) | ~3.0 | 0 | 5.29/3.71 | **falsified**: VNIR pixel = 0.21 nm; a global sub-pixel tolerance deletes Al and Fe entirely. Tolerance must stay resolution-aware (per-λ), not be globally tightened to cal-rmse |
| veto band 0.033→0.020 @shift 0 | 0.000 | 14.80 | 2.49 | 0 | 4.036 | **falsified**: vetoes real Fe (pixel-quantization scatter) while Sn still passes — a single per-element band cannot separate lucky-coherent FPs from quantization-scattered real elements |

## Ranked fix proposals

1. **Kill (or bound and re-objective) the post-calibration mop-up scan.**
   After a quality-passed calibration (`rmse_nm` ≈ 0.029 here), set
   `residual_shift_scan_nm = 0`, or at most `±2·rmse` with the scan objective penalizing
   |shift| (prefer the smallest shift within one F1 unit, not exact-tie only).
   Principle: a fitted axis must not be re-broken by a match-count-greedy global offset;
   measured edge-seeking in 3/3 configurations.
   Expected: Al 19.4→~10.8, Mg kept (1.7→2.7), RMSE 4.50→2.35 on loc1; Mn 0.44→0 (it was
   3× over-attributed shift-scoop); **must be paired with an FP guard** (see 3) because
   the well-centered axis lets Sn cohere by chance (2.45 wt%).
   Validate: harness on loc1–4 + the synthetic identifier benchmark (project memory:
   benchmark-gate identifier-scoring changes) + gate criterion (c) no confounder FPs.

2. **Gate the affine segment model on inlier *coverage*, not just count/rmse.**
   Accept `affine` for a segment only if the inlier span covers (say) ≥70 % of the
   segment and the implied edge extrapolation beyond the inlier hull is < 1 pixel;
   otherwise degrade to `shift` (the existing sparse-segment mechanism, currently keyed
   on point count only, at `sparse_segment_points=400`). Principle: never extrapolate a
   dispersion slope past its anchors; the VNIR slope is anchored at 475–650 nm and is
   wrong by ~0.2 nm at 877–906 nm against the Ca II triplet.
   Expected: correct 877 doublet member (Al −0.6 wt% further), Ca II 854 recovered,
   892-region mismatch moves out of reach of Al I 892.356 even with a small residual
   scan; E-probe (shift-only VNIR + no mop-up) measured RMSE 2.253, Al 10.20, Mg 2.71.

3. **Per-LINE residual gating at observation build (element-coherent match validation).**
   The veto already demands per-element coherence; the surviving elements still *carry*
   their incoherent lines into the Boltzmann fit. After computing the consensus residual,
   drop individual matched lines with |residual − consensus| > band (band = current
   tol/3, kept resolution-aware per-λ — NOT a global tightening, which probe C falsified).
   At shift 0 this drops 877 (res −0.064) and the 0.096-residual match while keeping
   394/396/308/309: Al →~3.9–8.6 wt% depending on whether 877 is rescued by fix 2 first.
   It is also the only probe-supported lever against the Sn-type lucky-coherent FP
   *line sets* (6 Sn lines spread across the Fe/Ti UV forest): per-line gating plus a
   per-peak ownership rule (a peak inside a kept element's match set cannot double-count
   as evidence for a weaker claimant — extend `used_peaks` arbitration from f1-order to
   expected-emissivity order) removes the FP mass without touching real majors.
   Validate: loc1–4 harness with confounders on (criterion c), synthetic ID benchmark F1.

Not recommended: global tolerance tightening (C/F), veto-band tightening (measured Fe
deletion), or defaulting calibration off (row C of the harness bridge: fragile, deletes
Mg/Mn, catastrophic without confounders — row F RMSE 11.0).

### Residual physics note (out of scope here)

Both axes leave the 877-doublet point ~+3 ln above the resonance-anchored Boltzmann
line, and cal-off's "exact" 6.88 wt% relies on it (without any 877 line Al falls to
~3.9). Likely a blend/self-absorption imbalance (resonance anchors optically thick →
intercept biased low; the unresolved IR doublet integrates both members' flux). Any fix
that removes the contaminated lines should re-baseline Al against this effect (SA mode
`observable`, or fitting the doublet as combined gA) before judging the final Al number.

## Raw artifacts

- `/tmp/rc-al-calon.log`, `/tmp/rc-al-caloff.log`, `/tmp/rc-al-{calon,caloff}-nostark.log`
- `output/bhvo2_measure/rc-al-{calon,caloff,calon-nostark,caloff-nostark}.json`
- `/tmp/rc_lines_{calon,caloff}.json` (per-match dumps), `/tmp/instrument_lines.py`,
  `/tmp/instrument_veto.py`, `/tmp/leverage.py`, `/tmp/fixprobe.py`
