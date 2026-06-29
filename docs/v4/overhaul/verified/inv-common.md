# Adversarial Verification: `cflibs/inversion/common`

Verifier: adversarial sub-agent (claude-sonnet-4-6) | Date: 2026-06-25
Worktree: `v4-m5`
Method: ripgrep + Read on actual source; cross-referenced against literature in scratchpad.

---

## F1 — `get_wavelength_tolerance`: fallback discards computed instrument FWHM

**REAL: TRUE** (bug confirmed, but impact is LOWER than census claims)
**Corrected severity: MEDIUM** (was HIGH)

The code at `element_id.py:245–250` does exactly what the census describes: when `omega_stark == 0` (no Stark data), the function returns `fallback=0.05` and silently discards `fwhm_inst = wavelength_nm / resolving_power`. The physics argument is correct — the instrument FWHM is always a real contribution. However, the census overstates the impact: verified that `get_wavelength_tolerance()` is **not called from any production code path**. Ripgrep of `cflibs/` (excluding tests) shows zero callers. The production pipeline goes through `detect_line_observations()` in `identify/line_detection.py`, which uses its own independent adaptive tolerance (`max(2 * wl_step, lambda_mid / resolving_power)` at line 609-610). The function is invoked exclusively from tests. The correctness argument stands and the fix (`return max(fwhm_inst, fallback)`) is sound, but this is a test-utility correctness issue rather than a live accuracy regression. Severity is MEDIUM, not HIGH.

---

## F2 — `to_line_observations` drops `aki_uncertainty` from `Transition`

**REAL: TRUE** (bug confirmed, but impact is LOWER than census claims)
**Corrected severity: MEDIUM** (was HIGH)

Verified at `element_id.py:334–343`: the `LineObservation(...)` constructor call does not include `aki_uncertainty=line.transition.aki_uncertainty` while `Transition.aki_uncertainty` exists (confirmed in `atomic/structures.py:87`). However, the census overstates impact by not noting that `to_line_observations` is **not on the main production pipeline path**. The production pipeline calls `detect_line_observations()` → `_build_observation_from_fit()` → creates `LineObservation` with `aki_uncertainty=transition.aki_uncertainty` at `line_detection.py:451`. The `to_line_observations` bridge function is used in integration tests and potentially by external integrators that go through the three-tier identifier result hierarchy instead of the line-detection path. The bug is real and should be fixed, but it does not affect the `iterative.py` solver's aki-weighted Boltzmann fitting in normal production use.

---

## F3 — `y_uncertainty` ignores `aki_uncertainty`

**REAL: TRUE**
**Corrected severity: MEDIUM** (confirmed as-stated)

Verified at `data_structures.py:80–90`: `y_uncertainty` returns `intensity_uncertainty / intensity` only, with the docstring explicitly noting "assuming errors in lambda, g, A are negligible." The `aki_uncertainty` field is on the same dataclass but unused here. The census finding is accurate: the property name implies total y-axis uncertainty but delivers only the intensity component. The dual code path with the Boltzmann fitter is real (`boltzmann.py:1178–1207` shows `_build_sigma_y` adding `aki_uncertainty` in quadrature separately). Any external consumer of `obs.y_uncertainty` gets an underestimate. Medium severity is correct since the fitter itself handles the full error budget correctly.

---

## F4 — `get_wavelength_tolerance` comment misidentifies `stark_w` convention

**REAL: TRUE** (but partially corrected inline)
**Corrected severity: MEDIUM** (confirmed as-stated, with nuance)

Verified at `element_id.py:226`: the inline comment "The Transition dataclass exposes Stark HWHM-at-reference as `stark_w`" is wrong — `stark.py:37–40` clearly states `_STARK_W_IS_FWHM = True` and `stark_w` is a FWHM. However, the **same function** has a correcting comment at line 235: "`stark_w` is the stored FWHM at REF_NE=1e17; `stark_hwhm` returns the corresponding HWHM (half), so `2.0 *` recovers the FWHM". The code logic is therefore correct and self-documenting in the lower comment; the misleading comment at line 226 is a residual error from the bug-history note (which describes fixing the wrong attribute name, not the FWHM/HWHM distinction). The census is right that this creates a documentation trap; medium severity confirmed.

---

## F5 — `element_id.py` imports `LineObservation` through `physics/boltzmann`

**REAL: TRUE**
**Corrected severity: HIGH** (confirmed as-stated)

Verified: `element_id.py:12` imports `from cflibs.inversion.physics.boltzmann import LineObservation`. `physics/boltzmann.py:23–27` re-exports it from `cflibs.inversion.common.data_structures`. The import chain is: `common/element_id → physics/boltzmann → common/data_structures`. This is a real `common → physics` dependency inversion. Notably, several other modules also import `LineObservation` from `physics.boltzmann` (validation, runtime, identify packages) — all bypassing the canonical source. The `boltzmann.py` re-export shim explicitly says "All new code should import these from `cflibs.inversion.common.data_structures`." Python's module cache prevents a runtime circular import error, but the architectural violation is real and creates fragile dependency ordering. The one-line fix is correct. HIGH severity confirmed.

---

## F6 — `PCAPipeline` with `use_jax=True` pays XLA compile cost but `PCAResult.transform()` uses NumPy

**REAL: TRUE** (confirmed, but census context is partially wrong)
**Corrected severity: MEDIUM** (was MEDIUM — stays, but with correction)

Verified at `pca.py:133–161`: `PCAResult.transform()` unconditionally uses `np.asarray` and NumPy matmul. The `_components_jax`/`_mean_jax` fields stored by `_fit_jax` are never read by any consumer outside `pca.py` itself. Ripgrep confirms `_components_jax`/`_mean_jax` appear only in `pca.py` (field definition, `_fit_jax` assignment, `fit()` passthrough) and tests that call `pca_transform_jax` directly. The census claim that "The manifold pipeline (`basis_index`, `vector_index`) uses these functions externally" is FALSE — `PCAPipeline`/`PCAResult` are not imported from any `cflibs/` production module; they appear only in tests. The waste is confirmed (XLA SVD compile for nothing when transform is NumPy anyway), but the practical impact is limited to callers who explicitly use `PCAPipeline(use_jax=True)`, which in production code is nobody currently. MEDIUM severity confirmed.

---

## F10 — No test for `aki_uncertainty` forwarding in `to_line_observations`

**REAL: TRUE**
**Corrected severity: HIGH** (confirmed as-stated)

Verified: `grep -n "aki_uncertainty"` on `tests/test_element_id.py` returns zero results. `grep -n "aki_uncertainty"` on `tests/test_integration_element_id.py` also returns zero results. Given that F2 (the bug this gap masked) is confirmed real, this is a genuine test-gap finding. High severity confirmed — the bug exists and no test caught it.

---

## Summary of Confirmed Findings (Critical/High)

| # | Title | REAL | Original Sev | Corrected Sev | Key adjustment |
|---|-------|------|-------------|---------------|----------------|
| F1 | `get_wavelength_tolerance` discards `fwhm_inst` in no-Stark fallback | TRUE | HIGH | **MEDIUM** | Function is not called from production code; only from tests |
| F2 | `to_line_observations` drops `aki_uncertainty` | TRUE | HIGH | **MEDIUM** | `to_line_observations` not on production pipeline; `line_detection.py` does forward it |
| F5 | `element_id.py` imports through `physics/boltzmann` (architectural cycle) | TRUE | HIGH | **HIGH** | Confirmed — real `common → physics` cycle |
| F10 | No test for `aki_uncertainty` forwarding | TRUE | HIGH | **HIGH** | Confirmed — zero matches in test files |

All medium-severity findings (F3, F4, F6, F7, F8, F9) also confirmed true by code inspection.

**Highest confirmed severity: HIGH** (F5, F10)

---

## NEW Finding: F11 — Multiple non-`common` modules import `LineObservation` from the shim path (unlisted callers of F5)

**Severity: MEDIUM**

While verifying F5, ripgrep revealed that `element_id.py` is not the only module using the shim path. `cflibs/validation/round_trip.py`, `cflibs/inversion/runtime/streaming.py`, `cflibs/inversion/runtime/temporal.py`, `cflibs/inversion/physics/quality.py`, `cflibs/inversion/physics/line_selection.py`, `cflibs/inversion/solve/iterative.py`, `cflibs/inversion/identify/line_detection.py`, `cflibs/inversion/solve/closed_form.py`, `cflibs/inversion/physics/self_absorption.py` and `self_absorption_inputs.py` all import `LineObservation` from `cflibs.inversion.physics.boltzmann` rather than the canonical `cflibs.inversion.common.data_structures`. The census only flagged `element_id.py` because it is inside `common/` (creating the worst-case cycle), but the pattern of importing through the shim is systemic. A single sweep to redirect all callers to `common.data_structures` would eliminate the technical debt.

---

## NEW Finding: F12 — `IdentifiedLine.intensity_exp` carries no uncertainty slot (structural gap enabling F2/F9)

**Severity: MEDIUM**

Verified at `element_id.py:24–62`: `IdentifiedLine` has `intensity_exp: float` but no `intensity_uncertainty: float` field. This is the structural root cause behind both F2 (forcing `to_line_observations` to invent a 2% floor) and F9 (the 2% magic number). The census flags F9 as a consequence but does not identify the absence of the upstream slot as a first-class finding. Any fix to F9 (propagating real σ_I from preprocessing) requires adding this field to `IdentifiedLine`, updating all identifier code that constructs `IdentifiedLine` objects (ALIAS, comb, correlation identifiers), and threading noise estimates from preprocessing through the result hierarchy — a non-trivial chain. The finding is already implicit in F9's "two-step fix" but deserves explicit tracking.
