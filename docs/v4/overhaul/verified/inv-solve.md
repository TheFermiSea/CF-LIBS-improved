# Adversarial Verification: `cflibs/inversion/solve`

Verifier model: claude-sonnet-4-6  
Date: 2026-06-25  
Source census: `scratchpad/overhaul/census/inv-solve.md`  
Method: rg + Read on actual code at `.worktrees/v4-m5`; re-derived physics against `literature/saha-boltzmann-lte.md`

---

## F1 — Saha-correction missing per-element `ln(U_II/U_I)` in ionic-line y-shift

**REAL: FALSE**  
**Corrected severity: none (no bug)**

The census derives that the y-shift for an ionic line onto the neutral plane must include `-ln(U_II/U_I)`. This is incorrect. Re-deriving from first principles:

- `y_ion = const + ln(n_II) - ln(U_II) - E_k_ion/(kT)`
- Saha: `ln(n_II) = ln(n_I) + ln(SAHA_CONST/n_e × T_eV^1.5) + ln(U_II/U_I) - chi/(kT)`
- Substituting: `y_ion = const + ln(n_I) + ln(SAHA_CONST/n_e × T_eV^1.5) + ln(U_II/U_I) - ln(U_II) - chi/(kT) - E_k_ion/(kT)`
- Simplify `ln(U_II/U_I) - ln(U_II) = -ln(U_I)`:
  `y_ion = [const + ln(n_I) - ln(U_I)] + ln(SAHA_CONST/n_e × T_eV^1.5) - (chi + E_k_ion)/(kT)`

So `y*_ion = y_ion - ln(SAHA_CONST/n_e × T_eV^1.5)` maps the ionic line to the same form as a neutral line with `x* = E_k_ion + chi`. The `U_II/U_I` term cancels exactly — it appears from both the Saha numerator and the Boltzmann denominator. The code at `iterative.py:1330-1331` is mathematically correct. The SB-graph docstring at line 1486 also correctly states "partition functions are absorbed into the fitted neutral intercept." The census confused itself by not substituting Saha into the level-population expression before cancellation.

---

## F2 — Lax path `n_e` update uses only 1-atm pressure balance; Stark diagnostic silently dropped

**REAL: TRUE**  
**Corrected severity: high**

Confirmed at `iterative.py:1617`: the condition `not diags` means that supplying Stark diagnostics while `use_lax_while_loop=True` forces the Python path, silently defeating the performance benefit of the lax path. The lax body at lines 850-858 contains only pressure-balance logic — there is no Stark pathway wired into the JAX kernel. The docstring comment at line 1609-1611 acknowledges this behaviour but there is no user-facing warning when the combination is used. The pressure-balance n_e update is physically non-standard for LIBS plasmas (Tognoni 2010 / Ciucci 1999 recommend Stark diagnostics as the canonical n_e probe). The severity is real but the fallback is graceful (it uses the Python path which IS correct) — so this is a performance and usability issue, not a correctness regression.

---

## F3 — `two_region` T_corona = 0.8 × T_core magic number: no literature basis, wrong element scope

**REAL: TRUE**  
**Corrected severity: high**

Confirmed at `iterative.py:1888-1893`. The code itself says "it has NO specific literature attribution (it is not a Hermann 2017 value)" in the inline comment. The 0.8 factor and the 0.3/0.7 weighting at line 1166 are undocumented empirical choices. The element set `{"Si", "Fe", "Ca", "Al", "Mg"}` at line 1158 has no physics rationale in the code or docs — hydrogen, sodium, and other species are treated differently without explanation. The lax path explicitly defers corona weighting (line 841: "spec §11: corona-element weighting deferred"). This is a real physics concern: the `two_region` flag introduces a per-element systematic Saha multiplier bias that is neither literature-grounded nor benchmark-gated. However, the severity should be qualified: `two_region=False` is the default, so this path is not exercised unless explicitly opted into.

---

## F4 — Gaussian likelihood default in Bayesian sampler is Pearson-biased; `poisson` mode is off-by-default

**REAL: TRUE**  
**Corrected severity: medium**

Confirmed at `bayesian/likelihood.py:34-82`. The code itself documents the Pearson bias on lines 51-54: "Putting the model prediction in the denominator biases the fitter toward under-estimating peaks (the classic 'Pearson' chi-square bias)." The Gaussian path (line 76-82) places `pred_safe` in the variance denominator; the Poisson (Cash 1979) path correctly separates the signal and denominator. The Gaussian default is retained "to avoid silently changing existing posteriors" (docstring line 66-68). For shot-noise-dominated LIBS spectra (ICCD detector), this is a genuine accuracy concern — peak amplitudes will be systematically underestimated. The severity is medium rather than high because the Bayesian path is an opt-in advanced feature (the main iterative solver is not affected) and the Poisson path is available and documented.

---

## F5 — `CFLIBSResult` defined inside `iterative.py` but consumed as shared result type

**REAL: TRUE**  
**Corrected severity: high**

Confirmed. `CFLIBSResult` is defined at `iterative.py:94`. Downstream consumers importing it:
- `cflibs/inversion/solve/closed_form.py:34`: `from cflibs.inversion.solve.iterative import CFLIBSResult`
- `cflibs/io/exporters.py:41`: `from cflibs.inversion.solve.iterative import CFLIBSResult`
- `cflibs/jitpipe/pipeline.py:213`: `from cflibs.inversion.solve.iterative import CFLIBSResult`

The census claim is confirmed and understated — it's not just `closed_form.py` and `physics/matrix_effects.py` but also `jitpipe/pipeline.py`. This creates a hard coupling: any refactoring of `iterative.py` requires coordinated changes in `io/`, `jitpipe/`, and `solve/`. Since `CFLIBSResult` is the result contract for ALL solvers, it belongs in `cflibs/inversion/common/`.

---

## F6 — `saha_boltzmann_graph=True` silently disables JAX lax path without a parity test

**REAL: TRUE**  
**Corrected severity: high**

Confirmed at `iterative.py:1617`: `if HAS_JAX and self.use_lax_while_loop and not self.saha_boltzmann_graph and not diags`. Setting `saha_boltzmann_graph=True` with `use_lax_while_loop=True` silently runs the Python path. No warning is emitted (confirmed via `rg -n "saha_boltzmann_graph.*warn"` returning empty). The test file `tests/inversion/solve/test_saha_boltzmann_graph.py` (252 lines) tests only the SB-graph path in isolation — it has no comparison of `saha_boltzmann_graph=True` vs `False` on the same observations. The docstring at `iterative.py:1006-1028` describes the SB-graph mode in detail but does not mention the lax path incompatibility. This is a genuine usability and correctness-assurance gap.

---

## F7 — `solve_with_uncertainty` re-runs Saha correction + fit after `solve()` — double partition-function queries

**REAL: TRUE**  
**Corrected severity: medium**

Confirmed at `iterative.py:2760-2782`. After `self.solve()` completes (line 2731), `solve_with_uncertainty` explicitly calls `_apply_saha_correction` (line 2763), `_evaluate_partition_functions` (line 2766), and re-fits via `_fit_saha_boltzmann_graph` or `_fit_common_boltzmann_plane` (lines 2779-2782). These involve redundant SQLite queries and O(n_lines) Python object allocation. The converged state from `solve()` is available but the intermediate structures (corrected observations, partition functions) are not stored. The re-fit itself is conceptually necessary (the uncertainty quantification needs the fit covariance at the converged T/ne), but the re-evaluation of partition functions and Saha correction is purely redundant. This is a real performance issue in batch pipeline mode where `solve_with_uncertainty` is called per-spectrum.

---

## F8 — `_apply_saha_correction` allocates new `LineObservation` objects per ionic line per iteration

**REAL: TRUE**  
**Corrected severity: medium**

Confirmed at `iterative.py:1338-1359`. Every call to `_apply_saha_correction` creates new `LineObservation` dataclass instances for ALL lines (ionic and neutral), not just ionic ones — neutral lines at lines 1349-1358 also allocate new objects unnecessarily. The census understated this: neutral lines create copies even though no value changes. At ~10 iterations for a 100-line fit, this is O(1000) Python object allocations per solve call, each triggering a `defaultdict(list)` append and subsequent GC pressure. The fix (precomputed numpy arrays) is straightforward and would eliminate all per-iteration allocation. However, profiling data from `reference_inversion_hotspot_profile.md` shows the dominant cost is the RANSAC wavelength calibration (73%) — this allocation fix is real but low-ROI on the total pipeline time.

---

## F9 — Module-level env-var sentinel functions wrap constructor argument reads redundantly

**REAL: TRUE**  
**Corrected severity: medium**

Confirmed at `iterative.py:35-90` and lines 939-945. Three module-level functions (`_jax_boltzmann_composition_enabled`, `_lax_while_loop_enabled`, `_reliability_from_uncertainty_enabled`) each wrap a single `os.environ.get()` call. They are called only from the constructor. The constructor itself has the caller-override logic (`use_jax_boltzmann is None`). This is genuine flag-debt: three separate functions each providing one layer of indirection for a single env-var read. One helper or inline reads would suffice. However note the inconsistency: `_reliability_from_uncertainty_enabled` (line 79-90) accepts "1", "true", "yes", "on" while `_lax_while_loop_enabled` (line 59) only accepts "1" — a subtle behavioral inconsistency across the three related env vars.

---

## F10 — Missing parity tests for SB-graph vs common-slope; HybridInverter manifold stale; lax `ne_from_stark` path untested

**REAL: TRUE**  
**Corrected severity: high**

Confirmed:
- `tests/inversion/solve/test_saha_boltzmann_graph.py` (252 lines) contains no test comparing `saha_boltzmann_graph=True` vs `False` on the same observations — confirmed by `grep -n "saha_boltzmann_graph=False"` returning empty.
- `tests/test_hybrid_inversion.py` exists but its backend fixture depends on a stale manifold (per project memory).
- `iterative.py:2529` hardcodes `ne_from_stark=False` in the lax path quality metrics (not read, but the lax path can never produce `ne_from_stark=True` by construction — see F2). No integration test asserts the lax→Python fallback behaviour when Stark diagnostics are supplied.

---

## Additional Findings Not in Census

### FA1 — Env-var inconsistency: `_lax_while_loop_enabled` only accepts "1", while `_reliability_from_uncertainty_enabled` accepts "1/true/yes/on"

**Severity: low**

`_lax_while_loop_enabled` (line 59) tests `== "1"` only. `_reliability_from_uncertainty_enabled` (lines 85-90) accepts `{"1", "true", "yes", "on"}`. `_jax_boltzmann_composition_enabled` (line 44) tests `== "1"` only. This means `CFLIBS_USE_LAX_WHILE_LOOP=true` silently does nothing while `CFLIBS_RELIABILITY_FROM_UNCERTAINTY=true` works. Users who follow the standard `true/false` shell convention for the lax env-var will silently get the Python path. This is a usability bug, not a physics bug.

### FA2 — `_apply_saha_correction` also copies neutral lines (no value change), doubling allocation vs what F8 describes

**Severity: low** (reinforces F8)

Lines 1349-1358 create new `LineObservation` objects for neutral (stage != 2) lines with unchanged field values. The census F8 description noted "O(n_ionic_lines) allocations" but the actual code allocates for ALL lines (ionic + neutral). For a typical Fe spectrum with 80% neutral lines, F8 underestimates the allocation by ~5×.

### FA3 — Lax path corona weighting explicitly deferred (line 841), creating a documented divergence from the Python path when `two_region=True`

**Severity: low** (subissue of F3, but independently confirmed)

The lax body comment at line 841 says "spec §11: corona-element weighting deferred" and uses `T_K` uniformly for all elements. The Python path applies `T_saha = 0.3*T_K + 0.7*T_corona` only to `corona_sensitive` elements. When `two_region=True` and `use_lax_while_loop=True`, the lax path would actually give different Saha multipliers than the Python path. However, since `two_region=True` + `use_lax_while_loop=True` is an unusual combination (both are opt-in), this is low impact in practice.

---

## Verification Summary

| Finding | REAL | Confirmed Severity |
|---------|------|--------------------|
| F1 | **FALSE** | none — math is correct, U_II/U_I cancels |
| F2 | TRUE | high |
| F3 | TRUE | high |
| F4 | TRUE | medium |
| F5 | TRUE | high (broader than census: 3 importers confirmed) |
| F6 | TRUE | high |
| F7 | TRUE | medium |
| F8 | TRUE | medium |
| F9 | TRUE | medium |
| F10 | TRUE | high |
| FA1 (new) | TRUE | low |
| FA2 (new) | TRUE | low (amplifies F8) |
| FA3 (new) | TRUE | low |

**Highest confirmed severity: high** (F2, F3, F5, F6, F10)

The single False finding (F1) is the most consequential correction: the census's proposed "fix" of adding `ln(U_II/U_I)` to the y-shift would actually INTRODUCE a physics bug, since the partition function ratio cancels algebraically in the Saha-Boltzmann plane derivation. No change should be made to `_apply_saha_correction`.
