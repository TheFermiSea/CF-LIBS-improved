# w3 diagnostic ID corpus (ak3.2.0) — provenance pointer

Regenerated diagnostic synthetic ID corpus that fixes the two structural
confounds proven in `docs/audit/2026-06-15-id-f1-rootcause.md`:

1. **Window** — moved from the deep-UV iron-group line forest (224–265 nm,
   ideal-detector F1 ceiling ~0.31) to the vis-NIR (385–850 nm), where all 10
   non-Fe panel elements have >=50% genuinely Fe-separable strong lines
   (measured via `scripts/analyze_window_separability.py`).
2. **Panel hygiene** — every panel element is now exercised. The old ak3.1.3
   corpus left Al/Co/Cu/Mg/Si/Ti/V in no recipe; the diagnostic set puts every
   element in >=4 recipes (Mg=4 min, Fe=12 max).

## Location (untracked — large blobs)

`data/synthetic_corpus/w3_diagwindow_v1/` (gitignored — regenerable):
- `corpus.json` (~24 MB), `corpus.h5` (~6.7 MB)
- `manifest.json`, `manifest.jsonl`, `summary.json`

The ak3.1.3 baseline (`data/synthetic_corpus/corpus.json`, name
`w2_fixedforward_v1`) is left untouched as the comparison baseline.

## Build invocation (deterministic, seed 42)

```bash
PYTHONPATH=$PWD JAX_PLATFORMS=cpu python scripts/build_synthetic_id_corpus.py \
  --db-path ASD_da/libs_production.db \
  --output-dir data/synthetic_corpus \
  --dataset-name w3_diagwindow_v1 \
  --version ak3.2.0 \
  --recipe-set diagnostic \
  --seed 42 \
  --elements "Fe,Ni,Cr,Mn,Cu,Ti,Si,Al,V,Mg,Co" \
  --lambda-min 385 --lambda-max 850 --pixels 2560 \
  --temperature-range-eV 0.8,1.8 --log-ne-range 16.3,18.0 \
  --snr-db 30,40 --continuum-level 0.03 --resolving-power 700,1000 \
  --shift-nm="-1.0,1.0" --warp-quadratic-nm 0.0
```

Builder change: commit `88e816e4` (`diagnostic_recipes()` + window/version/
max-spectra CLI hooks). Builds on the noise-overestimation fix `28f0e8b4` and
stratified-sampling reporting `a800919f`.

## Realized corpus facts (verified)

- `n_spectra` = 184 (23 diagnostic recipes x 8 perturbations, 8 per recipe)
- Realized wavelength axis: 384.0–849.0 nm, 2560 px, step ~0.182 nm
  (the forward model clips/rounds the requested 385–850 nm endpoints)
- Perturbation axes: SNR {30,40} dB, continuum 0.03, RP {700,1000},
  shift {-1.0,+1.0} nm, warp 0.0 → 8 combos/recipe
- Present-element union == panel exactly: Al Co Cr Cu Fe Mg Mn Ni Si Ti V (11/11)
- Per-element recipe coverage: Al 8, Co 5, Cr 8, Cu 8, Fe 12, Mg 4, Mn 11,
  Ni 10, Si 9, Ti 6, V 5
- Physics spot-check (`pure_Cr_0000`, T~9610 K, ne~1.3e17, shift -1.0 nm):
  all 7 top peaks (396, 424, 426, 428, 449, 520, 529 nm) map to genuine Cr DB
  lines within <=0.38 nm (5/7 within 0.3 nm; the two larger residuals are the
  -1.0 nm shift perturbation against RP=700 grid quantization) — physics-sane.

Expected F1 ceiling lift vs ak3.1.3: 0.31 -> 0.65–0.80 (re-measure after a
benchmark run with `scripts/benchmark_synthetic_identifiers.py`).
