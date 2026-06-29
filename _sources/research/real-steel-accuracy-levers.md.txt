# Real-Data Composition-Accuracy Levers (literature-grounded)

**Status:** active exploration (all levers green-lit 2026-06-27). Gate = real-steel benchmark
(`data/real_steel/steel_266.parquet`, PhdYoda/steel_266_LIBS, 36 NIST-class steel samples,
certified Cr/Ni/Fe/Si/Cu/Mn/Mo/C wt%, 4096-ch spectra 194-948 nm, 15 shots each).

## Root cause of the real-steel failure (diagnosed + literature-confirmed)

Naive constrained solve over-estimates trace minors (Cu 0.19 wt% recovered as ~93%,
"keystone collapse"). Diagnosis (`scripts/_realsteel_diag2.py`):
- Cu observed **only via singly-ionized (Cu II) lines** (0 neutral / 3 ion), high E_k 8.2-8.8 eV.
- Fitted plasma T low (~6705 K vs expected 8000-12000 K).
- Saha back-correction ion->total: `N_I ∝ N_II · N_e · exp(E_ion/kT)`. At low T, `exp(E_ion/kT)`
  explodes -> the solver infers a massive unobserved-neutral Cu population -> Cu soaks the closure.

**NotebookLM grounding (notebook `f1d2a053`, sources: CF-LIBS review; Tognoni/Cristoforetti
2010 review; Zhao et al. 2018 Plasma Sci. Technol. 20 035502 copper-lead alloy):**
1. *Ion-only over-estimation is a known failure mode* — under-estimated T makes exp(E_ion/kT)
   blow up the unobserved-neutral population (exactly our Cu).
2. *Composition is hypersensitive to T.* Zhao 2018: Pb-only lines -> 6700 K, Cu lines -> 9300 K
   (mirrors our 6705 K from a low-E_k/ion-biased fit); relative errors 12-32% at wrong T vs
   1.8-13.4% at the optimal T (7110 K) found by minimizing the composition standard error.
3. *Line selection:* cover a WIDE E_k range; neutral-only lines bias T low; avoid resonance /
   self-absorbed lines (or use Columnar-Density Saha-Boltzmann, Cristoforetti-Tognoni 2013);
   prefer lines with small self-absorption coefficient (e.g. Cu I 312.61 nm).
4. *Absolute closure flaw:* a ~1% error on the major matrix element -> huge relative error on
   minors. **One-Point Calibration (OPC, Cavalcanti 2013):** one matrix-matched standard fixes
   an empirical F(λ) correction applied to the Boltzmann plot -> much better minor accuracy
   while keeping CF-LIBS' T/n_e adaptivity. **This is the known-matrix (steel/Ti-alloy/DED) case.**

## Lever set (each real-steel-gated; implement as opt-in, parity-safe)

| # | Lever | Mechanism | Expected impact |
|---|-------|-----------|-----------------|
| L1 | **Optimal-T / OPC δ(T,α) minimization** (Zhao 2018) | scan T (+α) to minimize composition error vs a matrix-matched standard | HIGH — composition is hypersensitive to T |
| L2 | **Line-selection policy** | wide E_k spread, neutral anchor per element, drop ion-only-observed minors or require a neutral line, avoid self-absorbed/resonance | HIGH — direct cause of Cu trap |
| L3 | **OPC F(λ) correction** (Cavalcanti 2013) | one steel standard -> per-line multiplicative correction applied to Boltzmann plot | HIGHEST for known matrix (the real goal) |
| L4 | **CD-SB self-absorption** (Cristoforetti-Tognoni 2013) | columnar-density Saha-Boltzmann for self-absorbed major (Fe) lines | MED — Fe matrix self-absorption |
| L5 | **Real-axis wavelength calibration** | calibrate the measured axis before extraction | MED |
| L6 | **Saha ion-only robustness** | require neutral anchor / down-weight / flag ion-only-observed elements | MED — guards the Cu trap generally |
| L7 | **H_α / Stark n_e** | n_e from Stark width (self-absorption-free) instead of injected/pressure-balance | MED — n_e feeds Saha |

L1+L2+L3 directly target the diagnosed Cu/T failure and are the known-matrix (DED) sweet spot.
Combine winners; use the cluster for the combinatorial T/α/line-policy optimization, gated on
the 36-sample real-steel RMSEP (un-overfittable) + DED no-regression + NIST parity.
