# DED Benchmark — Noise & Accuracy Findings (2026-06-26)

Synthetic constrained-absolute benchmark on Ti-6Al-4V {Ti,Al,V}, nominal
T=11000 K, n_e=1e17, instrument FWHM 0.1 nm.

## Validated result: clean-floor solver accuracy

On a noise-free spectrum the constrained-element iterative solver (true n_e
injected) recovers the absolute composition well:

| element | truth | recovered | RMSEP over Al-scan |
|---|---|---|---|
| Ti | 90.0 | 88.3 | 1.39 |
| Al |  6.0 |  6.68 | 1.10 |
| V  |  4.0 |  4.99 | 0.79 |

So the **solver + clean extraction is accurate to ~1–2 wt% RMSEP** with a small,
stable (calibratable) bias. This is the trustworthy deliverable.

## Honest limitation: accuracy under DED noise

Under the (conservative, **guessed**) DED noise model (shot lognormal sigma=0.20,
Poisson SNR~100, readout 0.5% of peak, baseline 5%, T/n_e jitter), single-shot
recovery collapses (Ti bias ~-30 wt%) and **shot-averaging only partially
recovers** — Ti plateaus ~83 wt% at 50–100 shots, not the 88 clean floor.

Diagnosis (noise-source ablation): the bias is driven almost entirely by
**readout noise on weak lines**. Windowed integration with a positivity clip
*rectifies* readout noise and inflates weak minor-element lines (V, Al), pulling
the major element (Ti) down via closure. Removing the per-pixel clip (unbiased
integral + median-edge baseline) helps but a residual systematic bias survives
averaging, because the weakest high-E_k lines sit near the noise floor and the
log-Boltzmann step needs positive intensities (a selection bias). A pure-Gaussian
peak fit does not help here (it mismatches the Stark-broadened profile and biases
the clean floor).

## Implications / next steps

1. The **noise model must be calibrated to the real DED rig** — the noisy numbers
   are only as good as the assumed noise. The clean floor is model-independent.
2. Real DED-LIBS should **average many shots** and/or gate the detector for SNR.
3. **SNR-gated line selection** (drop lines below a per-shot SNR threshold) is the
   likely fix for the residual weak-line bias — to be added once real noise is known.
4. The stable clean-floor bias is **calibratable** against a known standard, which
   matters less for *drift* tracking (it cancels in measured − nominal).

## Multi-alloy clean-floor coverage (2026-06-26)

The benchmark + constrained solver generalize to the other AM alloys, with
element-specific accuracy that the benchmark now quantifies (clean floor, RMSEP wt%):

| alloy | best elements | weak element |
|---|---|---|
| Ti-6Al-4V {Ti,Al,V} | Al 1.1, V 0.8, Ti 1.4 | — (all < 1.5) |
| Inconel625 {Ni,Cr,Mo,Nb} | Nb 0.08, Mo 0.64 | Cr -4.5, Ni +5.1 |
| 316L {Fe,Cr,Ni,Mo} | Ni 0.12, Mo 0.60 | Cr -2.6, Fe +3.2 |

**Cr is systematically under-estimated** (~-3 to -5 wt%) across Inconel + 316L,
independent of resonance-line handling. Because the forward and solver share the
same atomic data, a persistent bias in this self-consistency test points to an
**ionization-stage gap**: line selection uses stages I/II, but at T=11000 K a
non-trivial fraction of Cr sits in Cr III, which the I/II inversion may not fully
recover via the Saha correction. UPDATE: ruled OUT partition data too -- Cr I/II/III all have real direct_sum_fit
partitions. With lines, resonance, stage-III, and partitions all ruled out, the
Cr bias is the iterative Boltzmann+Saha **method's inherent approximation error**
(element/condition-dependent), not a data bug. The converged full-spectrum (joint)
solver fits the forward directly (no Saha-correction approximation) and is the
likely improvement -- a future test on the synthetic benchmark.

## Iterative vs joint (full-spectrum) solver — Cr (2026-06-26)

Tested the converged full-spectrum (joint) solver on clean Inconel625 vs iterative:

| element | truth | iterative | joint |
|---|---|---|---|
| Ni | 64.5 | 69.3 | 56.5 |
| Cr | 22.5 | 18.3 | 24.3 |
| Mo |  9.5 |  8.9 | 13.8 |
| Nb |  3.5 |  3.6 |  5.3 |

The joint solver **fixes the Cr under-estimate** (24.3 vs iterative 18.3, truth 22.5),
confirming the Cr bias is the iterative Boltzmann+Saha *method* approximation, not a
data bug. BUT joint worsens Ni/Mo/Nb, so its overall RMSE on Inconel is higher
(~4.8 vs ~3.2 wt%). Conclusion: neither solver is uniformly best for these alloys;
they distribute error differently. A per-element or ensemble strategy (or improving
the iterative Saha extrapolation specifically for high-ionization transition metals)
is the research direction — not a one-line fix.

## SNR-gated line selection — counterproductive for minor elements (2026-06-26)

Tested dropping lines below a per-shot SNR threshold (extractor `min_snr`) on
noisy 10-shot Ti-6Al-4V:

| min_snr | Ti | Al | V | note |
|---|---|---|---|---|
| 0 (off) | 70.7 | 15.6 | 13.6 | noisy baseline |
| 5 | 86.6 | 13.4 | **0.0** | all V lines gated out -> V lost |
| 10 | 0 | 0 | 0 | all lines gated -> solve fails |

SNR-gating removes exactly the weak minor-element lines (V, low-concentration Al)
that we are trying to track -> the element drops to 0. For DED minor-element drift
tracking the answer to noise is **shot averaging** (reduces the noise floor so
weak lines stay measurable), NOT dropping low-SNR lines. The `min_snr` parameter
is kept (default 0 = OFF) for major-element-only scenarios but must NOT be used
when a tracked element is a weak minor constituent.

## Partition-data gaps for alloy elements (2026-06-26)

Energy-level counts in the DB (`energy_levels`, by `sp_num`) for the alloy
species — these drive the direct-sum partition fits:

| element | I | II | III |
|---|---|---|---|
| Ti | 202 | 101 | 1 |
| V  | 290 | 359 | 0 |
| Cr | 265 | 75 | 1 |
| Mo | 329 | **1** | 0 |
| Nb | **0** | **0** | 0 |

Implications:
- **Stage-III partitions cannot be fit** (V III, Mo III, Nb III, and effectively
  Cr III / Ti III from a single level) -> they use the generic U fallback. Low
  marginal impact at T~11000 K (small III fraction) but a real ceiling for
  high-T / high-ionization regimes.
- **Mo II has only 1 level** and **Nb has none at all** (Nb I/II partitions come
  from a stored polynomial, not levels) -> Mo/Nb Saha balance rests on weak
  partition data. This likely contributes to the Mo/Nb errors seen in Inconel625.
- FIX requires ingesting NIST ASD / Barklem-Collet 2016 energy levels for these
  species (a data task, network + DB migration) -- NOT fittable from the current
  DB. Flagged for the atomic-data backlog; gated on NIST ground truth.
