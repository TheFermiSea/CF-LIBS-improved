# Adversarial Verification: `cflibs/inversion/` Top-Level Modules

**Verifier:** Adversarial second pass against census `inv-top.md`
**Code revision:** v4-m5 main (2026-06-25)
**Method:** Independent ripgrep + Read at cited line numbers; re-derivation against
`full_spectrum.py` internals where the census made claims about data flow.

---

## Summary

Of the 4 critical/high findings (F1, F2, F3, F4), two are confirmed real bugs (F2, F3,
F4), one is a **false positive** (F1 — the conversion IS done inside
`solve_full_spectrum` before the return value; the census misread the data flow), and
one (F8, test-gaps) is confirmed. The highest confirmed severity is **critical** (F2).

---

## F1 — PHYSICS-CORRECTNESS — originally HIGH

**Title:** Full-spectrum solver adopts number-fractions as mass-fractions when the
iterative warm-start is superseded

**REAL: FALSE**

**Reasoning:** The census claimed that `concentrations=dict(fs.concentrations)` at
`pipeline.py:1219` passes raw number fractions. Independent re-reading of
`cflibs/inversion/solve/full_spectrum.py` refutes this.

- Line 647: `fit_mass = _number_to_mass_fractions(fit_numfrac)` — the number fractions
  from the JAX optimizer are converted to mass fractions inside `solve_full_spectrum`.
- Line 677–684: the adoption branch sets `adopted_conc = fit_mass` (mass fractions);
  the non-adoption fall-back sets `adopted_conc = dict(warm_mass)` where `warm_mass`
  comes from `warm_start_concentrations`, documented at line 405 as "MASS fractions (the
  iterative-solver / scoreboard convention)."
- Line 689: `concentrations=adopted_conc` — the `FullSpectrumResult.concentrations`
  field is therefore always mass fractions, regardless of whether the converged fit is
  adopted.

The census also incorrectly stated that `_number_to_mass_fractions` at "pipeline.py
line 949-971" is invoked for the "pure joint/bayesian path" but not the adopted branch.
In reality `_number_to_mass_fractions` in `pipeline.py` (line 949) is a helper used
only within `pipeline.py` for its own internal conversions; the full-spectrum solver
has its own copy in `full_spectrum.py` (line 131) and calls it before constructing
`FullSpectrumResult`. The adopted branch in `pipeline.py:1215-1227` correctly receives
and passes mass fractions through.

**Corrected severity: none** (not a bug).

---

## F2 — ARCHITECTURE/DESIGN — originally CRITICAL

**Title:** `ransac_early_exit` default in `build_pipeline_config` is `True` but the
dataclass field documents and defaults to `False` — hidden accuracy regression

**REAL: TRUE**

**Reasoning:** Independently verified at the cited lines.

- `pipeline.py:261`: `ransac_early_exit: bool = False` — dataclass field default is
  `False`. The inline docstring (lines 251-261) explicitly states "PARITY-AFFECTING on
  hard low-inlier cases" and "Default `False` reproduces the legacy loop."
- `pipeline.py:455`: `ransac_early_exit=bool(knob("ransac_early_exit", None, True))` —
  the `build_pipeline_config` fallback value is `True`.

The `knob()` helper resolves as: `config_overrides > config_yaml > default`. When
neither `config_overrides` nor the YAML provides a value, the third argument (`True`)
is used. This means every caller who goes through `build_pipeline_config` without
explicitly setting `ransac_early_exit` gets `True`, not the dataclass `False`. All
three entry points (`analyze`, `invert`, `batch`) go through `build_pipeline_config`.
The benchmark comment says this is "benchmark-gated" and parity-affecting, making this
a silent accuracy regression in production. The discrepancy cannot be caught at runtime
because both values are valid booleans.

No test exists for this consistency (confirmed: `rg ransac_early_exit tests/` returned
no results).

**Corrected severity: critical** (confirmed).

---

## F3 — ARCHITECTURE/DESIGN — originally HIGH

**Title:** `joint`/`bayesian` dispatch passes a fresh empty `diagnostics={}` dict to
`_run_full_spectrum_solver` — full-spectrum diagnostics are silently discarded

**REAL: TRUE**

**Reasoning:** Independently verified.

- `pipeline.py:1263-1264`: `_run_full_spectrum_solver(wavelength, intensity, atomic_db,
  pipeline, warm_start=warm, diagnostics={})` — the ephemeral `{}` is not connected to
  anything.
- `pipeline.py:1174-1188`: `_run_full_spectrum_solver` writes rich diagnostics
  (`full_spectrum_converged`, `fit_T_K`, `fit_ne_cm3`, `warm_start_concentrations`,
  `fit_concentrations`, all `diag_*` entries) into the `diagnostics` parameter it
  receives.
- `pipeline.py:1422`: `run_pipeline` calls `_dispatch_solver` without passing its own
  `diagnostics` dict as a parameter to `_dispatch_solver`. The function signature of
  `_dispatch_solver` (line 1230) has no `diagnostics` parameter.
- For iterative/closed_form paths the diagnostics flow correctly because
  `_run_peak_based_solver` does not write into a passed diagnostics dict; all
  diagnostics are accumulated in the `run_pipeline`-scope `diagnostics` dict directly
  (Stark results, observation counts, etc.). But for `joint`/`bayesian` the
  full-spectrum-specific entries (`fit_T_K`, `fit_ne_cm3`, convergence flags) are
  permanently lost.

The scoreboard and trust-report for `joint`/`bayesian` runs are completely blind to
whether the full-spectrum optimiser converged, what T/n_e it found, and whether the fit
was adopted. This is a real observability bug — not a correctness bug, but it makes
diagnosing poor joint/bayesian fits impossible from the returned diagnostics.

**Corrected severity: high** (confirmed).

---

## F4 — ARCHITECTURE/DESIGN — originally HIGH

**Title:** `__init__.py` `_ALIASED_EXPORTS` registers `ConvergenceStatus` pointing to
the Bayesian sub-package, silently shadowing the same name from `solve.joint_optimizer`

**REAL: TRUE** (partially — the census description of the mechanism is slightly
inaccurate but the net effect is real)

**Reasoning:**

Two separate `ConvergenceStatus` enums confirmed:
- `cflibs/inversion/solve/joint_optimizer.py:67`: `class ConvergenceStatus(Enum)`
- `cflibs/inversion/solve/bayesian/priors.py:170`: `class ConvergenceStatus(Enum)`

Mechanism (corrected vs. census): `_ATTRIBUTE_EXPORT_GROUPS` at line 183-190 registers
`"ConvergenceStatus"` from `cflibs.inversion.solve.joint_optimizer`. Then
`_ALIASED_EXPORTS` at line 217 registers `"ConvergenceStatus"` pointing to
`cflibs.inversion.solve.bayesian`. The `_MODULE_EXPORTS.update(_ALIASED_EXPORTS)` call
at line 225 **overwrites** the joint-optimizer entry with the bayesian one. Result:
`cflibs.inversion.ConvergenceStatus` resolves to the bayesian `ConvergenceStatus`
(from `priors.py`), while the joint optimizer's `ConvergenceStatus` is accessible only
as `JointConvergenceStatus` via the alias at line 216. The `__all__` list at line 351
includes `ConvergenceStatus` (bayesian) and `JointConvergenceStatus` (joint) with no
disambiguation comment.

The census claim that "the lazy loader races" is not quite accurate — this is a
deterministic dict overwrite, not a race. The real issue is the public-API ambiguity:
`from cflibs.inversion import ConvergenceStatus` silently gives the Bayesian enum;
callers checking `result.convergence_status` from a `JointOptimizationResult` get a
type they cannot directly compare against.

**Corrected severity: high** (confirmed; mechanism clarified).

---

## F8 — TEST-GAPS — originally HIGH

**Title:** No unit test for `ransac_early_exit` default-value consistency; no regression
guard for the mole→mass conversion in the adopted full-spectrum path

**REAL: TRUE** (partially — F1 is a false positive, so the second test guard is
unnecessary; the first is still missing)

**Reasoning:** `rg ransac_early_exit tests/` returns no matches. The test
`tests/cli/test_pipeline_defaults.py` exists but does not check
`ransac_early_exit` consistency between `AnalysisPipelineConfig()` and
`build_pipeline_config([])`.

The second guard (mole→mass regression for adopted full-spectrum path) is unnecessary
because F1 is false — `solve_full_spectrum` already returns mass fractions. However, a
test asserting that `build_pipeline_config` defaults match the dataclass defaults
(catching exactly the F2 regression) remains absent.

**Corrected severity: high** (confirmed for the ransac_early_exit consistency gap; the
mass-fraction round-trip test is unnecessary).

---

## Findings the Census MISSED

### M1 — CORRECTNESS — MEDIUM
**Title:** `_run_full_spectrum_solver` calls `solve_full_spectrum` with
`warm_start_concentrations=warm_concentrations` (line 1156), but `warm_concentrations`
is built at line 1143-1145 by directly iterating `warm_start.concentrations` — which is
a `CFLIBSResult` from the iterative solver. The iterative solver stores MASS fractions
in `concentrations`. Then `solve_full_spectrum` documents (line 405) that
`warm_start_concentrations` should be mass fractions. The round-trip is consistent.
However: `warm_concentrations` at line 1144 filters out `c <= 0.0` entries. If an
element has a zero mass fraction in the warm start but IS in `fit_elements` (line 1146),
it is excluded from `warm_concentrations` but included in the fit. Inside
`solve_full_spectrum`, `_mass_to_number_fractions` at line 159 uses
`max(float(mass_fractions.get(el, 0.0)), 1e-6)` for each element in the `elements`
list — so missing elements get a floor of `1e-6`. This is defensively correct but
means a zero-concentration warm start for one element biases the initial number-fraction
simplex point. Not a crash but a subtle warm-start quality degradation. Severity: low.

### M2 — ARCHITECTURE — LOW
**Title:** `_dispatch_solver` passes `uncertainty_mode` to `_run_peak_based_solver`
(line 1249) but does not pass it through to `_run_full_spectrum_solver` for
`joint`/`bayesian` solvers (line 1263). The full-spectrum result therefore always has
no uncertainty estimates, regardless of what `uncertainty_mode` was requested. Since
the full-spectrum solver does not currently compute uncertainties (it only provides
optimizer convergence diagnostics), this is not currently observable, but when
uncertainty support is added to the full-spectrum path the parameter-threading will
need to be added too. Severity: low (forward-compatibility note only).

---

## Dimension Summary

| Finding | Originally | Confirmed Real | Corrected Severity |
|---------|-----------|---------------|-------------------|
| F1 — mole/mass in adopted path | HIGH | **FALSE** | **none** |
| F2 — ransac_early_exit default mismatch | CRITICAL | **TRUE** | **critical** |
| F3 — lost full-spectrum diagnostics | HIGH | **TRUE** | **high** |
| F4 — ConvergenceStatus name collision | HIGH | **TRUE** | **high** |
| F8 — test gaps (default consistency) | HIGH | **TRUE (partial)** | **high** |
| M1 — zero-element warm-start floor | (missed) | new | low |
| M2 — uncertainty_mode not threaded | (missed) | new | low |

**Highest confirmed severity: critical (F2)**
