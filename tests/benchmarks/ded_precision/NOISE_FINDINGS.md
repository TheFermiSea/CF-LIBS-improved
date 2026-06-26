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
recover via the Saha correction. Candidate fixes (future work): include stage-III
lines / partition data for the high-ionization transition metals (Cr, V, Ti) and
verify the closure's Saha extrapolation accounts for unobserved stages. This is a
solver-physics item, gated on NIST/atomic-data ground truth, independent of the
benchmark scoring.
