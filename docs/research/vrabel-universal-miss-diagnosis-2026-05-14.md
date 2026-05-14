# Vrábel-2020 universal-miss root-cause diagnosis (2026-05-14)

## Summary

**Root cause: the alias identifier's `boltzmann_r2_min = 0.85` consistency
gate, driven by an unphysically cold plasma-temperature estimate
(~4000 K) returned by `_estimate_plasma_temperature` on Vrábel's
high-resolution echelle spectra.** A secondary contributor is the
`relative_cl_threshold = 0.1` element-floor gate, which removes Mg even
when its Boltzmann fit survives.

The five identifiers all share the same upstream pipeline (peak
detection → DB line lookup → matching). The lines ARE detected. The DB
DOES contain them. The matching DOES bind them to the right peaks. The
universal misses come from a downstream physics-consistency gate that
fires on every Vrábel spectrum but on essentially no BHVO-2 spectrum.

Concretely on sample s019 (true elements: Al, Ca, Fe, **Mg**, Pb,
**Si**; only Al detected by default):

| element | true | matched | k_sim | CL    | boltz_r² | detected? |
|---------|------|---------|-------|-------|----------|-----------|
| Mg      | yes  | 10/13   | 0.303 | 0.026 | **0.372** | **no** |
| Si      | yes  | 6/7     | 0.937 | 0.043 | **0.007** | **no** |
| Ti      | (no) | 7/17    | 0.382 | 0.005 | 0.577    | no |
| Al      | yes  | 4/4     | 0.942 | 0.133 | 0.956    | **yes** |
| Fe      | yes  | 16/19   | 0.877 | 0.102 | 0.724    | no |
| Ca      | yes  | 19/20   | 0.204 | 0.191 | 0.088    | no |
| Pb      | yes  | 8/9     | 0.979 | 0.130 | 0.595    | no |

Si I 288.16 nm is detected at **SNR=16**, Mg II 279.55 nm at **SNR=38**,
Mg I 285.21 nm at SNR=21. The peaks aren't hidden — the Boltzmann fit
slope is broken because the temperature estimate is wrong, the
expected line intensities used in the R² test are therefore wrong,
and the R² gate kills everything except Al (whose 4 matched lines
happen to be in a narrow E_k range where the broken-T emissivities
still roughly track observation).

## Hypotheses tested

### H1 (wavelength coverage) — **FALSIFIED**

Vrábel's echelle covers **200.000 – 1000.020 nm** uniformly at
**0.02 nm/px** across **40,002 channels**, no gaps. All canonical
LIBS resonance lines for the universally-missed elements are well
inside this range:

```
Si I 251.611, 288.158 nm           IN RANGE
Mg II 279.553, 280.270 nm          IN RANGE
Mg I 285.213 nm                    IN RANGE
Al I 309.271, 394.401, 396.152 nm  IN RANGE
Ti II 334.940, 336.121, 337.280 nm IN RANGE
```

`gaps > 0.1 nm: 0`. The spectrometer sees every line we would expect.

### H2 (atomic-database coverage) — **FALSIFIED**

`AtomicDatabase.get_transitions(...)` over Vrábel's 200–1000 nm range
returns the full canonical line lists:

| element | total lines (200–1000 nm) | ion 1 | ion 2 | strongest by A_ki·g_k |
|---------|---------------------------|-------|-------|-----------------------|
| Si      | 302 | 131 | 157 | Si I 251.61, 288.16, 252.41, 250.69 (all present) |
| Mg      | 364 | 171 | 189 | Mg II 279.55, 279.80, 280.27, Mg I 285.21 (all present) |
| Al      | 319 |  69 | 242 | Al I 309.27, 394.40, 396.15 (all present); Al II 219.26 |
| Ti      | 902 | 432 | 463 | Ti II 282.81, 334.94, 336.12, 337.28 (all present) |
| Cr      | 628 | — | — | full canonical |
| P       | 140 | — | — | full canonical |
| Pb      |  29 | — | — | full canonical |
| Cu      | 435 | — | — | full canonical |

The textbook resonance lines for every universally-missed element are
in the DB with finite A_ki, g_k, E_k, and Stark parameters.

### H3 (over-aggressive baseline subtraction) — **FALSIFIED**

Running `detect_peaks_auto(..., baseline_method=MEDIAN,
threshold_factor=3.0)` (exact bench config) on s019 returns **643
peaks** at noise σ ≈ 66. Residual-above-baseline at the strong
universally-missed-element lines:

```
Line                         I_peak    base   resid   SNR   peak<0.1nm?
Si I  288.158 (E_k=5.08 eV)   1063.6    -0.4  1064.0  16.1   1@288.180
Si I  251.611 (E_k=4.95 eV)    298.0    10.0   288.1   4.3   1@251.620
Mg II 279.553 (E_k=4.43 eV)   2558.9    25.9  2533.0  38.2   1@279.560
Mg II 280.270 (E_k=4.42 eV)   2215.5    27.5  2188.0  33.0   1@280.280
Mg I  285.213 (E_k=4.35 eV)   1413.8    11.7  1402.1  21.2   1@285.220
Al I  396.152                 3827.5   358.1  3469.4  52.4   1@396.160
Al I  394.401                 4023.2   374.8  3648.4  55.1   1@394.420
Al I  309.271                 1338.3    47.2  1291.1  19.5   1@309.260
Ti II 334.940                  719.7    60.8   658.9   9.9   1@334.940
Ti II 336.121                  438.2    55.5   382.6   5.8   1@336.140
```

The median baseline removes ~10–400 counts (the continuum) and leaves
the line cores well above the 3-σ threshold for every strong resonance
line. The peaks are present, the indices are matched, and they propagate
all the way into `ElementIdentification.matched_lines`.

### H4 (self-absorption) — **PARTIALLY TRUE**, but downstream of H_root

Self-absorption is plausibly real on s019 (Mg ~3 wt%, Al ~7 wt%, etc.)
— `_estimate_plasma_temperature` warns
``Positive or zero slope detected. Population inversion or error. T
set to infinity.``
which is exactly the signature of self-absorbed resonance lines
biasing a Boltzmann fit (low-E_k lines artificially weak relative to
high-E_k lines). The plasma-T estimator's fallback when this happens
is `self.reference_temperature = 10,000 K`, but the actual T returned
for s019 is **4,136 K** — so the fallback isn't kicking in either; the
fit converges on a cold-plasma minimum that explains the
narrow-line-strength range. Self-absorption is therefore the
upstream cause of the wrong T, not a directly fixable layer.

### H_root (boltzmann_r2_min gate × cold-T misestimate) — **CONFIRMED**

The default `boltzmann_r2_min = 0.85` requires that the matched lines
of an element fall on a Boltzmann slope with R² ≥ 0.85, using the
identifier's `_estimated_T`. When `_estimated_T` is wrong (e.g.
4136 K instead of the expected 7000–12000 K for soil LIBS plasma), the
predicted emissivities `g_k * A_ki * exp(-E_k / kT)` are wildly wrong,
the residuals are huge, and R² collapses. Almost every element fails
this gate. Al survives only because its 4 matched lines (308.21, 309.27,
394.40, 396.15 nm) all sit in a narrow E_k window (3.14–4.02 eV) where
the broken-T emissivities happen to track the observed intensities
closely (R² = 0.956).

Empirical confirmation — running the alias identifier on s019 with the
exact bench config except varying the two suspected gates:

```
Default (boltz_r2_min=0.85, rel_cl=0.1):    ['Al']
boltz_r2_min=0.0 (gate disabled):           ['Ca', 'Na', 'Si', 'Pb', 'Cu', 'Al', 'Fe']
boltz_r2_min=0.0 AND rel_cl_threshold=0.0:  ['Ca', 'Mg', 'Na', 'Si', 'Pb', 'Cu', 'Al', 'Fe']
rel_cl_threshold=0.0 only (boltz=0.85):     ['Al']        ← Boltzmann gate dominates
```

True elements on s019 = `{Al, Ca, Fe, Mg, Pb, Si}`. With both gates
relaxed we recover all 6 true elements (plus 2 false positives, Na/Cu).
**Boltzmann is the dominant filter; relative-CL adds the Mg kill.**

### Cross-dataset signature

Aggregating `estimated_T_K` and `effective_R` from the per-spectrum
annotations across all three datasets:

| dataset                       | rp_estimate | median T_K | T_K range  | median eff_R |
|-------------------------------|-------------|------------|------------|--------------|
| `aalto_libs`                  |        541 |    14,409  | 14,409 fixed |       541 |
| `bhvo2_usgs`                  |      9,433 |     6,595  | 6,588–6,827 |     5,837 |
| `vrabel2020_soil_benchmark`   |     30,001 |     4,257  |   3,295–10,533 |     7,092 |

Vrábel's plasma-T estimates cluster at ~4000 K — physically too cold
for LIBS by a factor of 2. BHVO-2 sits at a defensible 6,500–6,800 K.
The colder the T, the more sharply the gate fires.

Of the 16 Vrábel spectra with Si in the ground truth, **15 missed by
all 5 identifiers**. The one catch was s039, where the alias
identifier estimated T=5045 K and effective_R=10771 — both higher than
the typical Vrábel run — and `hybrid_union` (NNLS-based, doesn't use
the Boltzmann R² gate) detected Si. The other four identifiers still
missed it.

## Recommended fixes

In order of impact and tractability:

1. **Loosen / temperature-aware Boltzmann gate (highest impact).**
   Either:
   - Lower `boltzmann_r2_min` to ~0.5–0.6 by default (still rejects
     pure-noise matches without killing physically broken-T cases),
     OR
   - Skip the R² check entirely when `_estimated_T` is flagged as
     unreliable (population-inversion warning fired, or T outside
     5,000–15,000 K), OR
   - Fit T per element from the matched lines themselves and use
     the per-element R² rather than a globally-fixed T.

2. **Fix plasma-temperature estimation on high-R echelle data.**
   The current `_estimate_plasma_temperature` is biased toward 3,000–
   5,000 K on dense Vrábel-class spectra because too many weakly-emitted
   high-E_k lines outweigh the (self-absorbed) strong low-E_k lines in
   the Boltzmann slope. Options:
   - Restrict Boltzmann fit to lines with E_k spread > 2 eV.
   - Down-weight or exclude resonance lines (`is_resonance=1` in the
     DB) on samples where the species' major-element wt% > 1 %.
   - Switch to a Saha-Boltzmann two-population fit using lines from
     both ionization stages.

3. **Self-absorption correction (orthogonal but useful).** Once T is
   correct, multi-line self-absorption diagnostics
   (Karkare 2020, Aragón 2018) can flag optically-thick lines and
   correct their intensities before they enter the Boltzmann fit.

4. **`relative_cl_threshold` is too aggressive when one element dominates.**
   The Na CL=0.46 on s019 forces all other elements through a 0.046
   floor, dropping Mg (0.026). Consider:
   - Disable when `max_CL < 0.3` (low overall confidence regime), or
   - Apply per-ionization-stage instead of globally (Na I dominance
     shouldn't suppress Si I detection).

## Beads issues filed

See `## Filed bds` section below — three P1 items track the fixes.

## Reproduce

```bash
git -C /home/brian/code/CF-LIBS-improved worktree add /tmp/cf-libs-wt-vrabel-diag \
    chore/vrabel-coverage-diagnosis
cd /tmp/cf-libs-wt-vrabel-diag

# H1 + H2 + H3 + identifier reproduction (single self-contained script,
# matches the harness exactly):
#   - loads s019 / s064 / s039 from train.h5
#   - dumps wavelength range, DB line counts, peak SNRs at canonical lines
#   - runs ALIASIdentifier with default vs relaxed gates
#   - prints CL / boltz_r² / matched_lines per element
```

All numerical evidence in this report is reproducible from
`data/vrabel2020_soil_benchmark/train.h5`,
`ASD_da/libs_production.db`, and the alias identifier on the current
`dev` head (`c196a2e`).

## Filed bds

| bd ID | Priority | Title |
|-------|----------|-------|
| `CF-LIBS-improved-ftp1` | P1 | alias.boltzmann_r2_min=0.85 gate kills Si/Mg/Ti when plasma T mis-estimated on echelle data |
| `CF-LIBS-improved-762f` | P1 | _estimate_plasma_temperature returns ~4000 K on echelle soil spectra; expected 7000-12000 K |
| `CF-LIBS-improved-dj6y` | P1 | relative_cl_threshold=0.1 suppresses Mg when Na/Al dominate; should be per-ionization-stage or low-CL-regime aware |

`ftp1` is the surface-level fix (raise the floor or skip the gate
under cold-T conditions) and is the fastest path to recovering the
49 missed (spectrum, element) pairs.
`762f` is the deeper root cause (broken T estimate); fixing it makes
`ftp1` self-resolving without lowering the gate.
`dj6y` is the secondary Mg-only filter; without `ftp1` and `762f` it's
moot, but it should still be tightened.
