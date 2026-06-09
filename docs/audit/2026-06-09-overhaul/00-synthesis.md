# 2026-06-09 Overhaul Audit — Synthesis

Six parallel audits (forward physics, inversion solver, identification, pipeline defaults,
repo cleanliness, literature SOTA) of the full CF-LIBS pipeline, each grounded in
peer-reviewed literature and live measurements on real BHVO-2 ChemCam data. Detailed
findings with file:line evidence and citations live in `01-…06-` in this directory.
Tracking epic: `CF-LIBS-improved-zjg9`.

## Headline diagnosis

The codebase is *engineered* well (ruff/complexity clean, strong test count) but ships
**correctness bugs in the physics data path** and **defaults that bypass its own validated
improvements**. Measured on the same code, same spectra, same elements:

| Path | BHVO-2 RMSE (wt%) |
|---|---|
| `cflibs analyze` defaults | **10.29** (Fe=39 vs cert 8.6, converged=True, zero warnings) |
| `--saha-boltzmann-graph --closure-mode oxide` | **4.03** |
| Literature honesty check | no published *calibration-free* basalt result beats ~4; the 1.3–3.5 wt% figures are multivariate-calibrated (06 §Q5) |

So: 4.03 is already near CF-SOTA, the defaults throw away a 2.55× improvement, and the
remaining gap to sub-2 wt% requires *added physics* (self-absorption handled by default,
forward-fitting/two-zone, or one-point calibration), not tuning.

## Top correctness bugs (verified live)

1. **Forward model drops ~98% of lines** — populations keyed by raw float
   `level.energy_ev` (`plasma/saha_boltzmann.py:364`) but looked up with
   `round(E_k_ev, 8)` from the differently-sourced lines table
   (`radiation/emissivity.py:74`, `spectrum_model.py:38,215,247`): 110/6127 lines match.
   Poisons `cflibs forward`, NIST-parity scripts, and **all synthetic benchmark corpora**.
   (01-F1)
2. **U_II(Na)=15.0 hardcoded fallback** — DB has zero Na II/Li II/H II levels; the
   25/15/2 fallback ladder fires where true closed-shell U≈1.0. At 1 eV Na is ≳99%
   ionized → ~15× Na over-attribution: the original "Na=98%" blowup mechanism, still live.
   (02-F1)
3. **Neutral partition functions 5–40% low** (K I −40%, Na I −30%, Ca I −25% vs
   Barklem & Collet 2016) — `datagen_v2.py:167` regex misses high-Rydberg levels.
   Deflates alkalis/alkaline-earths, inflates Fe through Σ C=1. (01-F3)
4. **n_e never measured** — full Stark machinery exists, no path constructs a
   `StarkDiagnosticLine`; every solve uses the self-admittedly invalid 1-atm pressure
   balance. (02-F2)
5. **`total_species_density_cm3` silently ignored** in `radiation/kernels.py`
   (Bayesian forward): spectra bit-identical across a 10× density change. (05 §test-triage)
6. **Intensity floor unit-mixing** — `max(area, peak_height)` mixes counts·nm with
   counts (1.5–3 ln-unit Boltzmann distortions). (02-F6)
7. **Production element ID uses the worst identifier** — the comb-gate cascade
   (comb F1=0.012) gates `analyze`/`invert` while the benchmarked best, hybrid_union
   (F1=0.673), sits unwired. Single-line FP door admits Ag/Sn/W/Bi. (03 §1,3)
8. **`converged=True` on garbage** — the lax solve path omits the keystone gate and
   quality keys; no CLI path surfaces degeneracy/ne-fallback/dropped-element warnings.
   (02-F8, 04 §3)

## Structural insights

- **SB-graph ≡ common-slope under equal weights** (proved numerically to 1e-14): its gain
  comes from dropping w∝I weighting, so the weight cap is a band-aid; the principled fix
  is a total-error σ_y model with a systematic floor. (02-F7)
- **Self-absorption correction is a positive-feedback loop** (τ from recovered
  composition, no observable gate) — explains "SA makes intercepts worse". Fix:
  width-ratio / doublet-ratio gating (already implemented, unwired) or
  Völker & Gornushkin closed-form. (02-F4, 06 §Q2)
- **Five inconsistent line-strength metrics**, none using U(T); NIST `rel_int`
  (non-quantitative per NIST) is still load-bearing, including in the synthetic-truth
  forward model. (03 §5)
- **ps-LIBS (1 ps, 1040 nm)**: literature supports CF advantages (narrower lines, less
  SA) but a short early LTE window (~100–400 ns gate delays). No continuum model is
  defensible for gated ps acquisition. (06 §Q6, 01)

## Overhaul plan

**Wave 1 — correctness + defaults (beads z3cg, 16m7, cxxq, l4a8, +salvage):**
population matching fix; partition-function data integrity (B&C 2016 patch + fallback
ladder); intensity-floor removal + lax-gate parity + degeneracy honesty; best-path
defaults (geological preset = SB-graph + oxide), batch rewire, example-config fix, CLI
trust surface; salvage wt-pf g2/g4 test fixes → full suite green.

**Wave 2 — physics upgrades:** hybrid_union as production identifier (benchmark-gated);
Stark-n_e wiring; observable-gated self-absorption ON by default; spectral-response
E(λ) hook for the ps instrument; multi-line Boltzmann-coherence presence test;
total-error weighting; JAX Saha stage III + IPD; regenerate synthetic corpora post-W1.

**Wave 3 — cleanliness:** delete 7 legacy root scripts + 18 flat shims + ~100 stale
branches; consolidate remaining duplicate mass tables; archive one-off scripts; fix
CLAUDE.md falsehoods; re-baseline benchmarks.

**Gates (every wave):** BHVO-2 best-path RMSE ≤ 4.03 + per-element sanity; full-suite
diff vs baseline; ID benchmark for any identifier-scoring change; physics-only AST scan.
