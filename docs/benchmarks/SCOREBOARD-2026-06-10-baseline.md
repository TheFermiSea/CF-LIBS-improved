# CF-LIBS Goal-Metric Scoreboard — 2026-06-10 baseline

**One-command reproduction** (from the worktree root; per-spectrum detail lands in
`output/scoreboard_baseline/scoreboard.json` + `scoreboard.md`):

```bash
JAX_PLATFORMS=cpu PYTHONPATH=$PWD .venv/bin/python -m cflibs.cli.main scoreboard \
    --max-spectra 74 --seed 20260610 --output-dir output/scoreboard_baseline
```

(installed-entry-point equivalent: `JAX_PLATFORMS=cpu cflibs scoreboard --max-spectra 74 --seed 20260610`)

- Branch/commit: `overhaul/w4-scoreboard` @ `56e00b8` (off dev @ `abc049a`, the 2026-06 overhaul:
  fixed forward model, Stark n_e, calibrated axis, geological preset defaults).
- Pipeline: the PRODUCTION path (`cflibs.inversion.pipeline.build_pipeline_config` +
  `run_pipeline`), `geological` preset defaults — identical to `cflibs analyze`.
- Sampling: `--max-spectra 74 --seed 20260610` caps each dataset at 74 spectra with a seeded
  `np.random.default_rng` without-replacement draw. Effect on this run: bhvo2_chemcam ran all
  4/4, aalto ran all 74/74, synthetic_fixedforward ran a 74/288 sample (sampled ids recorded in
  the JSON artifact).
- Headline datasets are the REAL ones (bhvo2_chemcam, aalto). `synthetic_fixedforward` is our
  own forward model — track it for RELATIVE regressions only, never quote it as accuracy.

## Baseline reading (what these numbers say)

- **bhvo2_chemcam** — zero false positives (no Ag/Sn/W/Bi/Th confounder leaks) and median
  composition RMSE 2.52 wt% over the 10 certified elements. Recall 0.575: the misses are the
  same four minors/low-line elements on every location (K, Mn, Na, P; loc2 also drops Al) —
  the next accuracy lever is minor-element recovery, not major-element accuracy.
- **aalto** — precision 0.937 (6 FP over 74 spectra) but recall 0.471: mineral truth sets
  include minors the pipeline drops, and 11/74 spectra fail outright with "No usable spectral
  lines detected" (notably Pb and Ti pure-element standards). Failure-hardening on real spectra
  is a measurable lever here.
- **synthetic_fixedforward** — F1 0.236 with 46/74 failures. The corpus deliberately sweeps
  hostile perturbation axes (SNR down to 20 dB, wavelength shift up to -1.0 nm, quadratic warp,
  R~700); most failures are detection finding no usable lines under those perturbations, and
  the composition RMSE (43.9 wt% median) additionally reflects running the geological
  oxide-closure preset on metallic recipes (pure Fe/Ni, steels) — a basis mismatch by design,
  since the board always measures the production default. Use this row for relative
  robustness tracking between pipeline versions.
- **Runtime** — the whole pipeline is seconds per spectrum on CPU (median 2.7 s on ChemCam-sized
  spectra, ~0.8 s on Aalto, ~0.5 s on the synthetic band); wavelength calibration dominates
  (~60 % of wall), then detection+ID; the iterative solve is ~10 ms.
- **nist_srm_612 / nist_steel** — skipped-with-log: no public spectra ingested (the data
  directories are documented placeholders; see `data/nist_srm_612/README.md`).

Everything below this line is the verbatim generated artifact.

---

Generated: 2026-06-10T13:02:32+00:00  
cflibs: `/home/brian/code/CF-LIBS-improved/.worktrees/w4-scoreboard/cflibs`  
Preset: geological (production default)  
Sampling: seed=20260610, max_spectra=74

Reproduce:

```bash
JAX_PLATFORMS=cpu cflibs scoreboard --output-dir output/scoreboard --max-spectra 74 --seed 20260610
```

**Candidate-set policy:** candidates(spectrum) = truth.elements_present UNION confounder_elements (confounders: Ag, Sn, W, Bi, Th). **Presence rule:** solved mass fraction >= 0.005 (0.5 wt%). Failed spectra call nothing present (truth counts as FN) and are also listed under failures. ID metrics are micro-averaged; RMSE is the median per-spectrum RMSE over certified elements (element-wt% basis, oxygen excluded); runtimes are per-dataset medians.

| Dataset | Tags | Spectra (run/total) | P | R | F1 | RMSE wt% (med) | s/spectrum (med) | Failures |
|---|---|---|---|---|---|---|---|---|
| bhvo2_chemcam | geological, real | 4/4 | 1.000 | 0.575 | 0.730 | 2.52 | 2.7 | 0 |
| aalto | minerals, real | 74/74 | 0.937 | 0.471 | 0.627 | — | 0.8 | 11 |
| nist_srm_612 | placeholder, real | 0/0 (skipped) | — | — | — | — | — | — |
| nist_steel | placeholder, real | 0/0 (skipped) | — | — | — | — | — | — |
| synthetic_fixedforward | synthetic | 74/288 (sampled) | 0.583 | 0.148 | 0.236 | 43.92 | 0.5 | 46 |

## bhvo2_chemcam

> ChemCam (LANL testbed) LIBS of USGS BHVO-2 Hawaiian basalt; truth = USGS/GeoReM certified oxide composition converted to element wt% (cflibs.benchmark.reference_compositions.BHVO2_BASALT_USGS; oxygen excluded, oxide-bound). All 10 certified elements are above the 0.01 wt% presence cutoff. CCS spectra arrive response-corrected upstream; resolving power not certified (None).

- ID (micro): TP=23 FP=0 FN=17 — P=1.000 R=0.575 F1=0.730
- Runtime medians (s): total=2.68, calibration=1.57, detection+ID=0.90, stark n_e=0.15, solve=0.01
- Composition (n=4): RMSE wt% median=2.52, mean=2.71

| Element | mean signed error (wt%) |
|---|---|
| Al | -1.75 |
| Ca | -2.15 |
| Fe | +0.93 |
| K | -0.43 |
| Mg | -0.97 |
| Mn | -0.13 |
| Na | -1.65 |
| P | -0.12 |
| Si | +4.62 |
| Ti | +4.51 |

## aalto

> Aalto LIBS library pure-element standard (Al); presence-only truth. Nominal composition is ~100 wt% metal but element-wt comparison is skipped: the production geological preset closes on an oxide basis, which is ill-defined for a metallic standard. https://users.aalto.fi/~lainei1/pages/elements/

- ID (micro): TP=89 FP=6 FN=100 — P=0.937 R=0.471 F1=0.627
- Runtime medians (s): total=0.76, calibration=0.54, detection+ID=0.18, stark n_e=0.03, solve=0.01

Failures:

- `element_Pb`: ValueError: No usable spectral lines detected for inversion.
- `element_Ti`: ValueError: No usable spectral lines detected for inversion.
- `mineral_apatite41`: ValueError: No usable spectral lines detected for inversion.
- `mineral_hematite37`: ValueError: No usable spectral lines detected for inversion.
- `mineral_pentlandite331`: ValueError: No usable spectral lines detected for inversion.
- `mineral_pentlandite68`: ValueError: No usable spectral lines detected for inversion.
- `mineral_plagioclaseE2`: ValueError: No usable spectral lines detected for inversion.
- `mineral_pyriteE32`: ValueError: No usable spectral lines detected for inversion.
- `mineral_quartzE26`: ValueError: No usable spectral lines detected for inversion.
- `mineral_sideriteE40`: ValueError: No usable spectral lines detected for inversion.
- `mineral_wollastoniteE25`: ValueError: No usable spectral lines detected for inversion.

## nist_srm_612

**Skipped** — adapter yielded no spectra (see log / dataset README).

## nist_steel

**Skipped** — adapter yielded no spectra (see log / dataset README).

## synthetic_fixedforward

> SYNTHETIC corpus w2_fixedforward_v1 (version ak3.1.3) generated by OUR OWN fixed forward model — valid for RELATIVE comparisons between pipeline versions only, never a headline accuracy number (the generator shares physics with the inversion under test). Truth = generation recipe mass fractions; elements below 0.01 wt% excluded from elements_present. Source: /home/brian/code/CF-LIBS-improved/.worktrees/w1-integration/output/synthetic_corpus_w2/w2_fixedforward_v1/corpus.json

- ID (micro): TP=21 FP=15 FN=121 — P=0.583 R=0.148 F1=0.236
- Runtime medians (s): total=0.51, calibration=0.34, detection+ID=0.20, stark n_e=0.00, solve=0.00
- Sampled 74 of 288 spectra (seeded rng, seed=20260610).
- Composition (n=28): RMSE wt% median=43.92, mean=45.67

| Element | mean signed error (wt%) |
|---|---|
| Cr | +19.00 |
| Fe | -63.48 |
| Mn | -2.00 |
| Ni | -18.18 |

Failures:

- `pure_Fe_0001`: ValueError: No usable spectral lines detected for inversion.
- `pure_Fe_0004`: ValueError: No usable spectral lines detected for inversion.
- `pure_Fe_0007`: ValueError: No usable spectral lines detected for inversion.
- `pure_Fe_0009`: ValueError: No usable spectral lines detected for inversion.
- `pure_Fe_0013`: ValueError: No usable spectral lines detected for inversion.
- `pure_Fe_0020`: ValueError: No usable spectral lines detected for inversion.
- `pure_Fe_0023`: ValueError: No usable spectral lines detected for inversion.
- `pure_Fe_0043`: ValueError: No usable spectral lines detected for inversion.
- `pure_Fe_0045`: ValueError: No usable spectral lines detected for inversion.
- `pure_Fe_0057`: ValueError: No usable spectral lines detected for inversion.
- `pure_Fe_0060`: ValueError: No usable spectral lines detected for inversion.
- `pure_Fe_0061`: ValueError: No usable spectral lines detected for inversion.
- `pure_Ni_0004`: ValueError: No usable spectral lines detected for inversion.
- `pure_Ni_0006`: ValueError: No usable spectral lines detected for inversion.
- `pure_Ni_0010`: ValueError: No usable spectral lines detected for inversion.
- `pure_Ni_0017`: ValueError: No usable spectral lines detected for inversion.
- `pure_Ni_0021`: ValueError: No usable spectral lines detected for inversion.
- `pure_Ni_0024`: ValueError: No usable spectral lines detected for inversion.
- `pure_Ni_0029`: ValueError: No usable spectral lines detected for inversion.
- `pure_Ni_0031`: ValueError: No usable spectral lines detected for inversion.
- `pure_Ni_0032`: ValueError: No usable spectral lines detected for inversion.
- `pure_Ni_0033`: ValueError: No usable spectral lines detected for inversion.
- `pure_Ni_0037`: ValueError: No usable spectral lines detected for inversion.
- `pure_Ni_0041`: ValueError: No usable spectral lines detected for inversion.
- `pure_Ni_0071`: ValueError: No usable spectral lines detected for inversion.
- `binary_Fe_Ni_0001`: ValueError: No usable spectral lines detected for inversion.
- `binary_Fe_Ni_0016`: ValueError: No usable spectral lines detected for inversion.
- `binary_Fe_Ni_0021`: ValueError: No usable spectral lines detected for inversion.
- `binary_Fe_Ni_0025`: ValueError: No usable spectral lines detected for inversion.
- `binary_Fe_Ni_0041`: ValueError: No usable spectral lines detected for inversion.
- `binary_Fe_Ni_0049`: ValueError: No usable spectral lines detected for inversion.
- `binary_Fe_Ni_0051`: ValueError: No usable spectral lines detected for inversion.
- `binary_Fe_Ni_0053`: ValueError: No usable spectral lines detected for inversion.
- `binary_Fe_Ni_0054`: ValueError: No usable spectral lines detected for inversion.
- `binary_Fe_Ni_0065`: ValueError: No usable spectral lines detected for inversion.
- `binary_Fe_Ni_0071`: ValueError: No usable spectral lines detected for inversion.
- `steel_like_0007`: ValueError: No usable spectral lines detected for inversion.
- `steel_like_0012`: ValueError: No usable spectral lines detected for inversion.
- `steel_like_0018`: ValueError: No usable spectral lines detected for inversion.
- `steel_like_0021`: ValueError: No usable spectral lines detected for inversion.
- `steel_like_0023`: ValueError: No usable spectral lines detected for inversion.
- `steel_like_0029`: ValueError: No usable spectral lines detected for inversion.
- `steel_like_0032`: ValueError: No usable spectral lines detected for inversion.
- `steel_like_0038`: ValueError: No usable spectral lines detected for inversion.
- `steel_like_0048`: ValueError: No usable spectral lines detected for inversion.
- `steel_like_0062`: ValueError: No usable spectral lines detected for inversion.
