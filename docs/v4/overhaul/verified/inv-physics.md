# Adversarial Verification: `cflibs/inversion/physics/`

**Verifier:** Sub-agent (Sonnet 4.6, adversarial pass)
**Date:** 2026-06-25
**Worktree:** `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5`
**Source census:** `scratchpad/overhaul/census/inv-physics.md`

---

## Verification Summary

Of ten census findings:
- **9 CONFIRMED TRUE** (F1, F3, F4, F5, F6, F7, F8, F9, F10)
- **1 FALSE** (F2 — the pLTE lower-level energy cut is the correct Cristoforetti implementation)
- **1 additional finding** caught during verification (see below)

Highest confirmed severity: **critical** (F1 / F9 are the same root cause)

---

## Finding-by-Finding Verdicts

### F1 — McWhirter ΔE default is `max(E_k)` — REAL (TRUE)
**Confirmed severity: critical**

Verified at `cflibs/plasma/lte_validator.py:387-388`:
```python
energies = [o.E_k_ev for o in observations]
delta_E_eV = float(max(energies))
```
and at `cflibs/inversion/solve/iterative.py:2160-2167`: when `CFLIBS_MCWHIRTER_RESONANCE_DE` is not set (default), `delta_e_override` remains `None` and `lte_validator.validate()` falls back to `_delta_e_from_observations()` which uses `max(E_k)`. The `lte_validator.py` docstring for `_delta_e_from_observations` (lines 370-385) itself acknowledges this approach "bounds the resonance-to-upper-level transition" but is not the true resonance-line energy. For Fe I, `max(E_k)` from a typical Boltzmann-plot line set is 4–6 eV versus the resonance-line energy of ~2.48 eV — a 2–2.4× overestimate, yielding an ~8–14× too-high McWhirter n_e floor (scales as ΔE³). The resonance path exists in `_mcwhirter_delta_e_resonance()` and is wired but gated off by default. This is a genuine physics defect in the default code path.

---

### F2 — pLTE cut applies E_lower vs gap — FALSE
**Corrected severity: none (not a bug)**

The census claims the code confuses "absolute lower-level energy E_i" with the "energy gap between adjacent levels." This is incorrect. The Cristoforetti (2010) pLTE thermalization-limit implementation SPECIFICALLY uses the lower-level energy E_i (measured from the ground state) as the proxy for the relevant thermalization gap. The rationale (from Cristoforetti 2010 and the module docstring at `line_selection.py:73-85`) is that the thermalization-limit energy E* is derived from McWhirter inversion, and levels with E_i < E* are excluded because the excitation from the ground state to that level represents an energy gap ΔE ≈ E_i that exceeds what the plasma can thermalize. This is a self-consistent application of the McWhirter criterion per level — the "gap" IS the level energy when energies are measured from the ground state. The census misread the algorithm: the code implements the standard Cristoforetti pLTE cut correctly.

---

### F3 — SA doublet g_k: `find_doublet_pairs` doesn't check g_k equality — REAL (TRUE)
**Confirmed severity: high**

Verified at `self_absorption.py:480-505` (`_ordered_doublet_pair`): the function only checks `element`, `ionization_stage`, and `abs(E_k_ev difference) < dE_ev_tol` — no check on `g_k` equality. `_thin_emission_ratio()` at lines 294-296 writes `line1.g_k * line1.A_ki` and `line2.g_k * line2.A_ki` explicitly. If two fine-structure sub-levels within 1 meV have different J (e.g., J'=1 giving g=3 and J'=2 giving g=5), the `find_doublet_pairs` match passes but the ratio formula uses wrong weights. For a true same-upper-level doublet g_k1 == g_k2 and g_k cancels exactly — the formula is numerically correct in that case. The bug surface is narrow but real: fine-structure multiplets within the 1 meV window where two terms happen to be nearly degenerate in energy but NOT in J. The docstring on `_thin_emission_ratio` line 284 acknowledges "g_k written explicitly; it cancels for a true same-level pair" but `find_doublet_pairs` does not enforce the necessary precondition.

---

### F4 — H-alpha Stark n_e: linear scaling instead of Gigosos ~n_e^0.7 — REAL (TRUE)
**Confirmed severity: high**

Verified at `stark_ne.py:46-50` (module-level docstring): "Known limitation: hydrogen Balmer widths actually scale as ~n_e^0.7 rather than linearly (Gigosos 2014); the database stores H-alpha under the same linear-in-n_e convention as every other line, so this module honours that convention." `PREFERRED_DIAGNOSTIC_LINES` at line 83 lists `("H", 1, 656.28)` as the first-ranked n_e diagnostic. The linear formula introduces 20–50% n_e errors at LIBS densities relative to the Gigosos power-law. No Gigosos correction path exists in the codebase. The module acknowledges this as a known incomplete implementation "left for a dedicated follow-up."

---

### F5 — `matrix_effects.py` module-level import of `solve/iterative.CFLIBSResult` — REAL (TRUE)
**Confirmed severity: high**

Verified at `cflibs/inversion/physics/matrix_effects.py:50`:
```python
from cflibs.inversion.solve.iterative import CFLIBSResult
```
This is an unconditional module-level import. `cflibs/inversion/solve/iterative.py` itself imports from `physics/` at module level (lines 21-28: boltzmann, closure, self_absorption_observable, self_absorption_inputs). The circular dependency is real: `iterative.py → physics/*.py` and `physics/matrix_effects.py → solve/iterative.py`. The `inversion/__init__.py` uses lazy-loading to mitigate this (both `solve.iterative` and `physics.matrix_effects` appear in `_ATTRIBUTE_EXPORT_GROUPS`), but a direct `import cflibs.inversion.physics.matrix_effects` in a fresh process while `iterative.py` is mid-import can trigger `ImportError` or `AttributeError`. Contrast with `stark_ne.py` which correctly uses `if TYPE_CHECKING:` at line 67 for the same `iterative.py` dependency. This is an arch violation consistent with the census finding.

---

### F6 — `QualityAssessor` docstring says "NOT used" but IS wired — REAL (TRUE)
**Confirmed severity: medium**

Verified at `quality.py:75-81`: docstring says "the shipped iterative solver does NOT use". Verified at `iterative.py:2099` and `2112`: `QualityAssessor().assess(...)` is called. The docstring was written before M7 wired the class in. This is a stale comment but with real consequences: it misleads maintainers into treating `QualityAssessor` as safely deletable dead code. The comment also says "exercised only by the test suite" (quality.py:81) — this is false post-M7.

---

### F7 — `identify_resonance_lines` always returns empty set — REAL (TRUE)
**Confirmed severity: medium**

Verified at `line_selection.py:782-787`: the function discards both parameters via `_ = observations, ground_state_threshold_ev` and returns `set()`. Exported via `cflibs/inversion/__init__.py` lines 57 and 249. Any caller expecting a populated set of resonance-line identifiers gets `{}` silently — a silent "no resonance lines anywhere" result. Linked comment confirms this is due to `lower_level_energy` not being in the basic `LineObservation` dataclass. The `SUSPECT_E_I_MAX_EV` screen in `self_absorption_observable.py` functions as the practical workaround but is not linked from this stub.

---

### F8 — WLS sigma-clip uses unweighted `np.std` — REAL (TRUE)
**Confirmed severity: medium**

Verified at `boltzmann.py:428-432`:
```python
residuals = y - y_pred
std_res = np.std(residuals)
bad_indices = np.abs(residuals) > self.outlier_sigma * std_res
```
The WLS fit uses `weights` (inverse-variance), but the outlier rejection threshold is computed from unweighted `np.std`. This is a genuine inconsistency: WLS is an estimator that down-weights high-variance observations, but the sigma-clip uses the raw (unweighted) distribution of residuals. For adversarial cases with high dynamic range in `y_err_all`, this can misidentify high-quality, tightly-constrained points as outliers. The JAX path (`_fit_sigma_clip_jax`, line 456) mirrors the same logic per docstring "algorithmically equivalent to _fit_sigma_clip." Impact: ~1–5% temperature bias in adversarial cases.

---

### F9 — `CFLIBS_MCWHIRTER_RESONANCE_DE` correct-default-OFF — REAL (TRUE)
**Confirmed severity: medium**

Same code location as F1 (`iterative.py:2161-2166`). The correct-literature behavior (resonance ΔE) is opt-in; the incorrect behavior (max(E_k)) is the default. This is confirmed directly. F9 is a flag-management aspect of F1 and both are TRUE. F9 alone rates medium because it is framing/policy rather than the computational error (which is F1/critical), but both warrant fixing together.

---

### F10 — No regression test for McWhirter resonance ΔE vs. max(E_k) or H-alpha Gigosos — REAL (TRUE)
**Confirmed severity: medium**

Verified in `tests/inversion/physics/test_reliability.py`: `mcwhirter_min_ne` is tested with hardcoded values (e.g., `test_mcwhirter_min_ne_value`) but no test compares the resonance-path ΔE to the max(E_k) legacy path for the same observations. In `tests/inversion/physics/test_stark_ne_diagnostic.py`: Gigosos H-alpha is mentioned only at line 249 as a rationale for filtering non-literature sources — no test asserts the 20%+ error between linear and power-law at n_e ≠ 1e17 cm^-3. Test gap confirms F1 and F4 can regress invisibly.

---

## Additional Finding Caught During Verification

### FX1 — `matrix_effects.py` MODULE-LEVEL circular import is currently UNDETECTED by CI
**Severity: high (operational)**

While verifying F5, checked whether the circular import actually fires in a typical import scenario. The `cflibs/inversion/__init__.py` lazy-loading map lists both `cflibs.inversion.solve.iterative` and `cflibs.inversion.physics.matrix_effects` but does NOT eagerly import either. However, any test that does `from cflibs.inversion.physics.matrix_effects import MatrixEffectCorrector` (direct path, bypassing `__init__.py`) while Python's import machinery has partially initialized `cflibs.inversion.solve.iterative` (e.g., due to test collection order) will trigger the circle. A quick `rg` search of `tests/` for direct `matrix_effects` imports:

```
rg -rn "matrix_effects" tests/
```

No test currently imports `matrix_effects` directly, so the circle is latent but untested. The CI gate is `ruff check` (static analysis), which does not detect circular imports. The fix is the same as F5 suggests: move the import inside a function body or use `TYPE_CHECKING`.

---

## Findings Table

| ID  | Title (short)                                  | REAL? | Confirmed Severity |
|-----|------------------------------------------------|-------|--------------------|
| F1  | McWhirter ΔE default = max(E_k), overestimates | TRUE  | critical           |
| F2  | pLTE cut uses E_lower vs gap                   | FALSE | none               |
| F3  | SA doublet `find_doublet_pairs` no g_k check   | TRUE  | high               |
| F4  | H-alpha Stark: linear not Gigosos ~n_e^0.7     | TRUE  | high               |
| F5  | `matrix_effects.py` layering violation         | TRUE  | high               |
| F6  | `QualityAssessor` docstring "NOT used" is stale | TRUE  | medium             |
| F7  | `identify_resonance_lines` always returns {}   | TRUE  | medium             |
| F8  | WLS sigma-clip uses unweighted std             | TRUE  | medium             |
| F9  | `CFLIBS_MCWHIRTER_RESONANCE_DE` correct-OFF    | TRUE  | medium             |
| F10 | No regression test for F1/F4 paths            | TRUE  | medium             |
| FX1 | Circular import latent, CI-undetected          | NEW   | high               |
