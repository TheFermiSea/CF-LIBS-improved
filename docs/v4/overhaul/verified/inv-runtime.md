# Adversarial Verification: `cflibs/inversion/runtime/`

Verified against source at `.worktrees/v4-m5/cflibs/inversion/runtime/` (ripgrep + targeted Read)
and literature at `scratchpad/overhaul/literature/`.

---

## Verified Findings (critical/high only per task scope)

### F1 · CRITICAL · Optical depth formula: wrong degeneracy, wrong λ³ scaling, magic `1e-25`

**REAL: TRUE** — all three sub-errors confirmed.

`temporal.py:1014` sets `g_lower = g_k` with the comment "Approximate as similar to upper level",
where `g_k` is the upper level weight from `LineObservation`. The absorption cross-section scales
with the *lower*-level population `N_l g_u / g_l · exp(-E_l/kT)` (Eq. 8 in self-absorption-cog.md §1.4),
so substituting `g_k` (upper) for `g_l` (lower) introduces an off-by-`g_u/g_l` error in `n_lower`.
Line 1025 computes `tau = 1e-25 * A_ki * lambda_cm**3 * n_lower * L` — confirmed λ³ (should be λ²,
per the literature formula `q ≈ λ₀² * g_u * A_ul * N_s / (8π * Δλ_D * U_s(T)) * exp(-E_l/kT)`).
The docstring at line 993 says "concentration (mass fraction)" but line 1011 uses it as
`n_s = concentration * total_number_density_cm3` without molar-mass conversion — a dimensional
inconsistency confirmed by inspection. `SCALE_FACTOR = 1e-25` has no physical derivation in the
source or comments. Corrected severity: **CRITICAL** (unchanged).

---

### F2 · HIGH · `FastAnalyzer` placeholder returns `converged=True` with meaningless result

**REAL: TRUE** — both paths confirmed.

`streaming.py:710–722`: when `line_observations is None`, returns `CFLIBSResult` with hardcoded
T=10000 K, n_e=1e17, uniform concentrations, `converged=True`, `n_lines=0`. Comment says
"In production, would use line detection here / For now, return placeholder result" — this is
explicitly unimplemented. `streaming.py:785`: the real-physics path also hardcodes
`electron_density_cm3=1e17` with comment "Not determined in fast mode", but still returns
`converged=True`. Both paths actively mislead callers that check `result.converged`. The
`_process_single` method at line 1011 passes the result directly to `_assess_quality` and
`_collect_warnings` without distinguishing the placeholder path. Corrected severity: **HIGH** (unchanged).

---

### F3 · HIGH · SA score uses n_e as proxy for neutral absorber density — wrong direction in recombination

**REAL: PARTIALLY TRUE** — confirmed as a real imprecision, but the "wrong direction" claim
is overstated for the model used here.

`temporal.py:699–704` is confirmed to use `ne_ref / n_e` as the SA score, with no reference to
neutral density. The census argument is physically valid in a plasma with conserved total atoms:
during recombination, n_e decreases while neutral density increases — giving a spuriously optimistic
SA score. However, the `PlasmaEvolutionModel` (temporal.py:340–428) uses a purely exponential
decay for n_e with no ion/neutral partitioning. In this simplified expansion-dominated model, both
n_e and total density decrease monotonically, so the proxy is at least directionally consistent
for the dominant plasma dilution effect. The bug is real (n_e is a poor proxy for optical depth
and does not reflect the actual neutral column density that drives self-absorption), but the claim
that it is "monotone in the wrong direction" specifically during the recombination phase is model-
dependent. The fundamental error — using ionized electron density as a proxy for neutral absorber
density — is confirmed. Corrected severity: **MEDIUM** (downgraded from HIGH; the direction error
only manifests if recombination-phase physics is explicitly modeled, which this model does not do).

---

### A1 · HIGH · FastAnalyzer `exp(avg_y)` concentration proxy omits closure → physically wrong

**REAL: TRUE** — confirmed by reading both the proxy code and the y_value definition.

`streaming.py:767–768`: `avg_y = mean(y_value for lines of element)`, then
`concentrations[el] = exp(avg_y)`. From `data_structures.py:78`,
`y_value = ln(I * λ_nm / (g * A))`. After Boltzmann fitting, this quantity equals
`ln(F * C_s * hc/(4π) / U_s(T)) + slope * E_k` where `F` is the unknown experimental factor.
`exp(avg_y)` thus gives `F * (hc/4π) / U_s(T) * C_s * exp(slope * avg_E_k)` — not just `C_s`.
The normalization by `total` cancels `F` and `(hc/4π)` only if all elements have the same `U_s(T)`
and the same `avg_E_k`, which is generally false. This is a confirmed broken proxy for concentration;
the closure equation from `cflibs.inversion.physics.closure` is not called. Corrected severity:
**HIGH** (unchanged).

---

### C1 · HIGH · `SCALE_FACTOR = 1e-25`: vestigial magic constant with no physical derivation

**REAL: TRUE** — this is the same defect as F1 sub-error 2, confirmed at `temporal.py:1024`.

The constant `1e-25` appears as a local variable inside `optical_depth_at_time`, with comment
"f ~ A_ki * lambda^2 (rough scaling)" that does not justify the value. No units analysis is
provided. Given the correct formula (self-absorption-cog.md §1.4, Eq. 8) has a physically derived
prefactor `1/(8π * Δλ_D)`, the constant is unvalidated. Since this is the same code location as
F1, C1 is a duplicate framing of the F1 sub-error rather than a separate finding. Both are
**confirmed**. Corrected severity: **HIGH** (as standalone; but noting it overlaps with F1 CRITICAL
if considered together). The census correctly separates them — C1 is re-confirmed as a distinct
"flag debt" finding about the vestigial constant.

---

### T1 · HIGH · No physics parity test for `TemporalSelfAbsorptionCorrector`

**REAL: TRUE** — `TemporalSelfAbsorptionCorrector` has no callers in `cflibs/` other than
`temporal.py` itself and is referenced only in `tests/test_temporal.py`. A search confirms no
test checks that `f_tau ≈ 1.0` for optically thin lines or verifies the escape factor magnitude
against an analytically known tau. Given F1 breaks the tau formula, such a test would immediately
expose the magnitude error. Corrected severity: **HIGH** (unchanged).

---

## Downgraded or Modified Findings

| Census ID | Census Severity | Corrected Severity | Reason |
|-----------|-----------------|-------------------|--------|
| F3 | HIGH | MEDIUM | "Wrong direction" only manifests with explicit recombination phase; simplified exponential model makes n_e a consistent (not reversed) proxy for plasma dilution |

---

## Additional Findings Spotted During Verification

### X1 · MEDIUM · `PlasmaEvolutionModel` docstring vs implementation mismatch: power-law vs exponential n_e

**Location:** `temporal.py:354–358` (docstring), `temporal.py:428` (implementation)

The class docstring states two models for electron density: a power-law decay
`n_e(t) = n_e0 * (1 + t/t_0)^(-alpha)` and a "simplified exponential". The implementation
unconditionally uses the exponential form `n_e0 * exp(-t/tau_ne)`. No `alpha` or `t_0` parameter
exists in the constructor. This is a documentation-only inconsistency (no physics bug per se),
but a caller reading the docstring and expecting power-law behavior will be surprised. Severity:
**LOW** if documentation only, **MEDIUM** if power-law behavior is actually needed for physically
realistic late-time recombination modeling (the power-law better approximates recombination + expansion).

### X2 · LOW · `_correct_one_observation` docstring at line 1199 says "exactly as the inlined loop body did" — but this is a post-refactor correctness assertion not validated by any test

**Location:** `temporal.py:1197–1199`

The refactor extracting `_correct_one_observation` from an inlined loop body is claimed equivalent,
but no before/after snapshot test or round-trip test exists. Given the method has 13 parameters
including 5 mutable accumulators, a refactoring error would be silent. This is consistent with T1
(no parity tests) and reinforces the need for a correctness anchor. This is an addendum to A4 and T1,
not a separate root-cause.

---

## Summary Table

| # | Dim | Census Sev | Verified | Corrected Sev | Notes |
|---|-----|-----------|----------|--------------|-------|
| F1 | Physics | CRITICAL | TRUE | CRITICAL | All 3 sub-errors confirmed (wrong g, λ³, mass/number mismatch) |
| F2 | Physics | HIGH | TRUE | HIGH | Both placeholder and real-physics paths; `converged=True` is misleading |
| F3 | Physics | HIGH | PARTIAL | MEDIUM | Proxy is imprecise, but "wrong direction" claim is model-dependent |
| A1 | Arch | HIGH | TRUE | HIGH | exp(avg_y) confirmed wrong; missing closure equation |
| C1 | Complexity | HIGH | TRUE | HIGH | Same locus as F1; vestigial constant confirmed; noted as overlapping |
| T1 | Tests | HIGH | TRUE | HIGH | No parity test; escape factor untested against known tau |
| X1 | Docs | — | NEW | LOW/MEDIUM | Power-law docstring vs exponential implementation |
| X2 | Tests | — | NEW | LOW | Refactor equivalence unvalidated |
