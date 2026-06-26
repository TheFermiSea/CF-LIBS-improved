# ADJUDICATION — Adversarial Physics Audit of CF-LIBS

**Adjudicator role:** skeptic of the 11 domain skeptics, in both directions. Every "correct"
verdict was challenged for thin evidence; every "flawed" verdict was sanity-re-derived to rule
out a false alarm. Independent re-checks were run against the live code/DB at
`/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5` (provenance confirmed:
`cflibs.__file__` resolves inside the worktree).

---

## HONEST OVERALL VERDICT

**The codebase is NOT deeply flawed on physics fundamentals. It is largely sound, with a small
number of real, localized bugs — two of which matter on real data.**

The *core* CF-LIBS physics chain is correct and independently verified to machine precision:
Saha prefactor (0.08% off CODATA — cosmetic), Boltzmann-plot linearization, the Saha→neutral-plane
y-shift (including the correct cancellation of `ln(U_II/U_I)` that a *prior* audit had wrongly
flagged as missing), emissivity `ε=hc/4πλ·A_ki·n_k`, Doppler width, Voigt/Faddeeva, McWhirter
constant `1.6e12`, NIST line/A_ki parity (≤0.2%), ILR/Helmert isometry, oxide stoichiometry,
BIC/AIC/Gaussian/Cash likelihoods, and the *main* self-absorption module. These are not
rubber-stamps — they were re-derived with astropy/sympy and executed.

The flaws that DO exist are real and were correctly caught: a critical Stark H-alpha
reference-width error, a medium composition mole-vs-mass reporting defect with a peer-reviewed
citation (Völker 2024), incomplete partition-function atomic data for the primary analyte (Fe I),
and a dimensionally-broken self-absorption formula in a non-default runtime path. None of these
invalidate the underlying method; they are fixable bugs and data-completeness gaps.

So: **largely sound with specific, fixable bugs — not deeply flawed.** A prior audit that
rubber-stamped *everything* would have been wrong (the Stark/composition/partition flaws are
genuine), but an audit claiming the fundamentals are broken would *also* be wrong.

---

## CONFIRMED FLAWS (ranked by severity)

| # | Severity | File:line | Violates (literature) | Confirmed? | Fix | Benchmark-gate? |
|---|----------|-----------|------------------------|------------|-----|-----------------|
| 1 | **CRITICAL (conditional)** | `ASD_da/libs_production.db` H I 656.279 nm `stark_w=0.049 nm` `stark_w_source='stark_b'`; selected via `stark_ne.py:82` `PREFERRED_DIAGNOSTIC_LINES[0]` | Gigosos 2003 (*Spectrochim. Acta B* 58, 1489; 476 cites): H-alpha total Stark FWHM ≈1.3 nm at n_e=1e17, T=1e4 K — ion broadening dominates. Stored 49 pm is electron-impact ONLY (Konjevic 2002). | **YES** — DB value verified = 0.049/stark_b; Gigosos paper verified real. Re-derivation confirms ~27× deficit. | Remove H from `PREFERRED_DIAGNOSTIC_LINES` + exclude `H` in `measure_stark_ne()` until a Gigosos n_e^0.64 path exists; OR store 1.3 nm @ n_e^0.64 with a `gigosos` provenance class. | **YES** — n_e is an input to Saha; gate on real H-bearing spectra. |
| 2 | **HIGH** | `inversion/runtime/temporal.py:1024-1025` `SCALE_FACTOR=1e-25; tau=SCALE_FACTOR*A_ki*lambda_cm**3*n_lower*L` | Hutchinson *Princ. Plasma Diagnostics* eq 5.13 / Mihalas eq 4-2: correct prefactor `1/(8π^1.5·v_th)`≈9e-8 s/cm. 1e-25 is ~10^17–18× too small AND dimensionally wrong (s⁻¹·cm, not dimensionless). | **YES** — lines verified verbatim; dimensional analysis sound. Corrector returns τ~1e-16 always → permanent silent no-op. | Delegate to `SelfAbsorptionCorrector._estimate_optical_depth` (already correct) or insert the proper prefactor+Doppler φ(ν₀). | Yes if temporal-gate optimization is used on real data. NOT the default inversion path. |
| 3 | **MEDIUM** | `benchmark/scoreboard.py:303-305` reads `result.concentrations` (mole fractions, per `iterative.py:109`) ×100 as wt%; `pipeline.py:949 _number_to_mass_fractions` defined but **0 callers** | Völker 2024 (*J. Anal. At. Spectrom.*, DOI 10.1039/D4JA00028E): CF-LIBS closure yields MOLE fractions; omitting `w_s=C_s·M_s/ΣC_jM_j` gives up to +353% error. Tognoni 2010. | **YES** — verified scoreboard path, 0 callers of the pipeline-level converter, synthetic truth `true_composition=recipe.mass_fractions` (wt%). C-in-steel +362% reproduced. | Call `_number_to_mass_fractions` on iterative/closed-form results before scoring (full_spectrum path already does). Check no downstream consumer expects mole fractions. | **YES** — directly changes reported RMSE on steel/geological wt% truth. |
| 4 | **MEDIUM (data, not logic)** | `ASD_da/libs_production.db` energy_levels: Fe I=425 (NIST ~837, 51%), Ca I=76, Ti I=202; `partition_functions` BarklemCollet2016 rows=**0** | Barklem & Collet 2016 (A&A 588 A96); Irwin 1981 (ApJS 45 621). Incomplete U(T) → ~10–20% U(Fe I) undercount at ≥10⁴ K → Saha N_II/N_I bias. | **YES** — DB counts verified; 0 BC2016 rows verified. `partition.py` math itself is correct. | Run `patch_partition_functions_bc2016.py` / ingest complete NIST levels for Fe/Ti/Cr/Ca/Na/K; verify U(Ca I) error 25%→<1%. | **YES** — affects Fe (primary steel analyte) Saha correction. |
| 5 | **MEDIUM (non-default path)** | `inversion/identify/model_selection.py` `_component_line_strengths` (L1/σ) + `_compute_spic` (n·log(RSS/n)) | Webb et al. 2021 (*MNRAS* 501, 2268, DOI 10.1093/mnras/staa3551): R_a = Σ(c·B/σ)² (L2²) and data-fit = χ²=RSS/σ². | Plausible/likely — formulas verified present; not independently re-derived against Webb's full eq here. **SpIC is NOT default (BIC is).** | Use L2² R_a and χ² data-fit; pass `noise_variance` into `_compute_spic`. Or document as non-Webb approximation. | Only if `criterion=SPIC` is ever used. |

---

## "CORRECT" VERDICTS — CHALLENGED, UPHELD (well-evidenced)

These had BOTH a peer-reviewed/DB citation AND executed numerical proof. I re-ran the
load-bearing checks; they hold.

- **Saha prefactor** (`constants.py:76` 6.042e21): re-derived from astropy/CODATA = 6.03713e21,
  +0.0806% — cosmetic (≈5 K of 23,000 K). Factor-of-2 spin degeneracy correctly embedded
  (half would be 3.02e21). **Upheld, low.**
- **Boltzmann/Saha-plane inversion**: the `ln(U_II/U_I)` cancellation is algebraically real;
  the prior "missing term" alarm was the false one. Round-trip recovers T exactly. **Upheld, none.**
- **Emissivity / Doppler / Voigt**: emissivity matches astropy to 1e-12; Voigt = scipy wofz
  exactly. Only nit: `2.355` vs `2.35482` (0.0076%) in a non-production sizing path. **Upheld, low.**
- **NIST line parity**: 15 Fe I A/B+ lines ≤0.2% after the gA→A_ki/g_k correction; constants
  exact. **Upheld, none.**
- **McWhirter constant** `1.6e12`: exact vs 5 sources. **Upheld, low** (see weak-evidence note).

---

## "CORRECT/FLAWED" BUT WEAKLY EVIDENCED — NEEDS DEEPER CHECK

- **McWhirter ΔE definition (M2)** — verdict "correct/low", but the *default* path uses
  `max(E_k)` rather than the resonance ΔE that Cristoforetti 2010 §2.1 explicitly requires.
  The agent's defense ("conservative, false-negative-only, env-flag fix exists, off by default")
  is reasonable BUT the default ships a paper-NONfaithful threshold. I downgrade confidence:
  this is a **real deviation from the cited paper**, mitigated only by error direction and an
  off-by-default flag. Treat as **low-but-real**, not "correct." Recommend flipping the
  resonance-ΔE default (it is a documented memory item: `CFLIBS_MCWHIRTER_RESONANCE_DE`).
- **Stark severity is CONDITIONAL** — on synthetic round-trips the forward model ALSO uses the
  49 pm linear convention, so forward/inverse is self-consistent (the `stark_ne.py:46-50`
  docstring defense holds). The catastrophic 12–60× n_e error manifests ONLY on REAL H-bearing
  spectra. Because `stark_ne` is marketed as the real-data n_e diagnostic and H-alpha is the
  #1 preferred line, I keep **critical** but flag the conditionality explicitly — a deeper check
  on a real H-alpha LIBS spectrum would nail the in-production impact.
- **SpIC (#5)** — formulas present and the L1-vs-L2 / deviance-vs-χ² discrepancy is plausible,
  but I did not independently re-derive Webb's full Eq. 3/5 numerically here. Non-default, so
  low practical risk; mark **needs deeper check** before any SpIC production use.
- **VALD air/vacuum** — the agent itself concluded the *production* DB has zero VALD entries and
  the ingest is behaviorally correct; the finding is a wrong docstring + a latent footgun, not a
  live physics error. Correctly NOT counted as a shipped flaw. **Doc/process fix only.**

---

## BOTTOM LINE FOR THE USER

A prior audit that "rubber-stamped" the code would have missed flaws #1–#4, which are real.
But the user's stronger hypothesis — that the codebase is *deeply flawed on physics
fundamentals* — is **not supported by the evidence**. The governing equations are implemented
correctly and verified to machine precision. The genuine problems are: one critical reference-data
mismatch (H-alpha Stark, real-data only), one results-reporting unit bug (mole vs mass), one
atomic-data-completeness gap (Fe I levels), and one broken formula in a non-default runtime path.
All four are localized, citable, and fixable; all four should be benchmark-gated.
