# Adversarial Verification: cflibs/inversion/identify/

Verifier: Claude Sonnet 4.6 | Date: 2026-06-25
Method: ripgrep + direct Read on actual source + LineObservation.y_value tracing

---

## Summary of Verdict

| Finding | REAL | Corrected Severity | Notes |
|---------|------|--------------------|-------|
| F1 — Boltzmann y-axis inconsistency | **FALSE** | ~~HIGH~~ → NONE | Both paths use `obs.y_value = ln(I·λ/(g·A))` |
| F2 — CL gate contradicts paper 2-line support | **TRUE** | HIGH (confirmed) | Flows into detection via `_gate_relative_cl` |
| F3 — `_compute_ratio_consistency` returns 0.1 not 0.5 | **TRUE** | MEDIUM (confirmed) | Docstring/code mismatch; penalizes H-alpha |
| F4 — P_SNR recomputed in `_decide` | **TRUE** | MEDIUM (confirmed) | Redundant call confirmed at line 4659 |
| F5 — Thread-safety mutable state | **TRUE** | MEDIUM (confirmed) | Sequential risk is latent; concurrent risk real |
| F6 — JAX/CPU parity tests missing | **FALSE** | ~~MEDIUM~~ → NONE | `TestBuildTemplatesJax` exists in test_alias_jax_nnls.py |
| F7 — CPU LOO-NNLS O(N²) loop | **TRUE** | LOW (confirmed) | 12× np.delete + 12× NNLS confirmed |
| F8 — "weighted" mode dead production path | **TRUE** | LOW (confirmed) | Code comment says so explicitly |
| F9 — No regression test for CL=0 on 2-line elements | **TRUE** | MEDIUM (confirmed) | test_paper_faithful_contract.py doesn't cover it |
| F10 — Interference analysis lacks test | **FALSE** | ~~MEDIUM~~ → NONE | test_comb.py lines 145-179 and 398-410 cover it |

**Confirmed highest severity: HIGH (F2)**

---

## Detailed Findings

---

### F1 — REAL: FALSE

**Census claim:** Boltzmann y-axis inconsistency — temperature estimator includes λ but `BoltzmannPlotFitter` consistency check might not.

**Verification:**
- `alias.py:2511`: `y = math.log(I_obs * t.wavelength_nm / (t.g_k * t.A_ki))` — the legacy T-estimator builds y points manually.
- `BoltzmannPlotFitter.fit()` in `physics/boltzmann.py:256` uses `np.array([obs.y_value for obs in observations])`.
- `LineObservation.y_value` is a `@property` in `inversion/common/data_structures.py:78`: `return np.log(self.intensity * self.wavelength_nm / (self.g_k * self.A_ki))` — this INCLUDES `wavelength_nm`.
- The `_collect_boltzmann_observations` helper (alias.py:3135) builds `LineObservation` objects with `intensity=I_obs, wavelength_nm=trans.wavelength_nm, g_k=trans.g_k, A_ki=trans.A_ki`, so `obs.y_value` computes `ln(I·λ/(g·A))`.

**Both code paths use the identical y-axis: `ln(I·λ/(g·A))`**. The census assumed `BoltzmannPlotFitter` might use raw intensity — it does not; it delegates to `obs.y_value` which always includes λ.

---

### F2 — REAL: TRUE (HIGH confirmed)

**Census claim:** `_decide` zeros CL for N_matched < 3 (or N_matched < N_expected for N_expected ≤ 4), contradicting the paper's 1-line and sparse-line support.

**Verification:**
- `alias.py:4682`: `if N_matched < 3: CL = 0.0` — confirmed present.
- The comment at `alias.py:1863` states: "the hard N_matched>=3 gate … and the Boltzmann-R^2 gate are all homegrown and were removed." This is SELF-CONTRADICTORY; the gate was not actually removed.
- `detected = k_det > self.detection_threshold` at line 1866 is paper-faithful. However, `_gate_subset_relative_cl` at lines 1236–1241 later sets `e.detected = False` if `e.confidence < max_pool_max_confidence * threshold`. When CL is zeroed in `_decide`, `confidence=0` flows through `_apply_post_cl_gates` (which multiplies into 0), and this zero confidence eventually triggers `detected = False` in `_gate_relative_cl`.

**The full causal chain**: `_decide` zeros CL → `_apply_post_cl_gates` multiplies zero (unchanged) → `ElementIdentification.confidence=0` → `_gate_subset_relative_cl` sets `e.detected=False` because `0 < relative_threshold`. A 2-line element with a genuine k_det > detection_threshold can be demoted to detected=False purely from the N_matched < 3 gate, not from the paper's decision criterion. The comment claiming the gate was removed is wrong.

---

### F3 — REAL: TRUE (MEDIUM confirmed)

**Census claim:** `_compute_ratio_consistency` returns 0.1 for <3 matched lines, but docstring says 0.5 (neutral).

**Verification:**
- `alias.py:4426`: docstring says "Returns 0.5 (neutral) with < 3 matched lines."
- `alias.py:4430`: `return 0.1  # Penalize — too few lines for meaningful ratio check`
- `_apply_post_cl_gates` line 1983: `CL *= 0.5 + 0.5 * R_rat` → with R_rat=0.1, CL is multiplied by 0.55 (a 45% penalty). With the documented neutral 0.5, it would multiply by 0.75.

**Confirmed docstring/code mismatch**. Impact is attenuated: this only affects CL (confidence metadata), and CL=0 for N_matched<3 elements (from F2) makes this penalty redundant for the most affected elements. However, elements with exactly 3 matched lines that still pass F2's gate (N_expected>4) get a systematically lower CL from R_rat=0.1 even when the ratio check is simply uninformative. The fix (return 0.5) better matches the neutral-test semantics.

---

### F4 — REAL: TRUE (MEDIUM confirmed)

**Census claim:** `_dispatch_p_snr` is called twice — once at the outer loop and once inside `_decide` — with the second call ignoring the cached `global_p_snr`.

**Verification:**
- `alias.py:1344`: `global_p_snr = self._dispatch_p_snr(corrected_intensity, peaks)` — computed once per `identify()` call.
- `alias.py:4659`: `P_SNR = self._dispatch_p_snr(intensity, peaks)` — recomputed inside `_decide`, which is called once per candidate per phase.
- `_decide` signature takes `intensity` and `peaks` but NOT a pre-computed P_SNR. There is no code path that passes `global_p_snr` into `_decide`.
- For N=12 candidates × 2 phases (Phase 1 scoring loop + Phase 3 rescore when competition ran) = up to 24 redundant evaluations per `identify()` call.

**Confirmed redundant computation**. The fix is to add an optional `p_snr` parameter to `_decide`.

---

### F5 — REAL: TRUE (MEDIUM confirmed)

**Census claim:** `ALIASIdentifier` has mutable per-call state (`_effective_R`, `_global_wl_shift`, `_estimated_T`) that bleeds across sequential calls or creates hazards under concurrent use.

**Verification:**
- `alias.py:996–1003`: `self._effective_R`, `self._global_wl_shift`, `self._estimated_T` confirmed as instance-level mutable state.
- `alias.py:1310–1311`: `_sa_n_damped_lines` and `_sa_damped_elements` are reset at the top of `identify()`.
- `_estimate_plasma_temperature` at line 1336 ALWAYS runs and sets `_estimated_T`, so sequential calls are correctly handled in the happy path.
- Class docstring (line 626) does document the thread-safety limitation.
- The latent risk: if a hypothetical future code path skips `_estimate_plasma_temperature`, the stale `_estimated_T` from the previous call would corrupt `_compute_element_emissivities` (line 1515 passes `T_estimated=self._estimated_T`).

**Confirmed architecture debt**. Severity MEDIUM is correct — currently safe for sequential use, but the per-call context threading approach (option a from census) is cleaner.

---

### F6 — REAL: FALSE

**Census claim:** No parity tests cover the JAX template-builder path or the JAX Pearson-comb path.

**Verification:**
- `tests/inversion/identify/test_alias_jax_nnls.py:278`: `class TestBuildTemplatesJax` with methods including `test_jax_matches_cpu_rtol_1e_10` at line ~321 — CPU vs JAX template builder parity is TESTED with rtol 1e-10 tolerance.
- `tests/inversion/identify/test_comb_jax_correlate.py` exists, covering the JAX Pearson-comb path.
- The census assertion "lack dedicated parity regression tests" is FALSE.

---

### F7 — REAL: TRUE (LOW confirmed)

**Census claim:** `_compute_nnls_attribution` runs N leave-one-out NNLS calls in a Python loop — O(N²) at N=12.

**Verification:**
- `alias.py:4274–4282`: confirmed loop with `np.delete(A, j, axis=1)` per iteration and `nnls(A_reduced, peak_intensities)` per iteration.
- For N=12 candidates: 12 `np.delete` allocations (each creates a copy of the (N_peaks × 11) matrix) plus 12 NNLS solves.
- The JAX path (`compute_nnls_attribution_jax`) already vectorizes these via vmap and is the correct reference.

**Confirmed.** Severity LOW is appropriate — on typical LIBS spectra (N_peaks ~100, N_cands ~12) this is on the order of milliseconds total and not a dominant bottleneck.

---

### F8 — REAL: TRUE (LOW confirmed)

**Census claim:** `temperature_estimator_mode="weighted"` is a dead production path with no preset or benchmark coverage.

**Verification:**
- `alias.py:2474–2479`: code comment explicitly states it is "a Vrabel universal-miss investigation leftover that never graduated to a preset or sweep. It is exercised only by tests/test_alias_unit.py; production and benchmark presets use only 'legacy' (default) and 'robust'."
- The ALIAS_PRESETS dict at line 4704 has no preset that specifies `temperature_estimator_mode="weighted"`.

**Confirmed.** The code itself documents that this is dead production weight. Severity LOW is correct.

---

### F9 — REAL: TRUE (MEDIUM confirmed)

**Census claim:** No regression test for `_decide`'s CL=0 gate on 2-line elements.

**Verification:**
- `tests/test_paper_faithful_contract.py`: tests confirm `detected = k_det > C_th` (not CL-based), and that CL floors are banned. But there is NO test asserting `confidence > 0` for a 2-line element with `N_expected > 4`.
- `tests/inversion/identify/test_alias_presets.py`: no test for sparse element CL integrity.
- Given the interaction found in F2 (CL=0 → detected=False via `_gate_relative_cl`), the absence of a pinning test is a real gap. A future refactor that modifies the relative CL gate could silently cause false negatives on 2-line elements with otherwise-valid k_det scores.

**Confirmed test gap.** The test should assert both `confidence > 0` AND `detected = True` for a 2-line element with k_det > threshold, since the relative CL gate can override detection.

---

### F10 — REAL: FALSE

**Census claim:** CombIdentifier interference analysis (`_analyze_interferences`) has no test covering `_mark_reciprocal_interference`.

**Verification:**
- `tests/test_comb.py:145–179` (`test_analyze_interferences`): creates two elements with overlapping lines at < 0.1 nm, asserts `is_interfered=True` on both sides and that the non-overlapping tooth (`Fe at 500.0`) has `is_interfered=False`.
- `tests/test_comb.py:398–410` (`test_analyze_interferences_no_overlap`): separate test for well-separated lines → confirms no false interference marking.

**Both the positive and negative cases are covered**. The census finding is incorrect.

---

## Additional Findings Not in Census

### NEW-1 — LOGIC — MEDIUM
**Title:** `_gate_subset_relative_cl` raises `ValueError` if `max_pool` is empty

**Location:** `alias.py:1238`

```python
relative_threshold = max(e.confidence for e in max_pool) * threshold
```

`max()` on an empty iterable raises `ValueError: max() arg is an empty sequence`. The callers at lines 1228–1233 guard with `if neutrals:` / `if ionized:` / `if unclassified:`, but the `unclassified` case passes `all_element_ids` as the `max_pool`. If `all_element_ids` is empty (e.g., no elements pass Phase 1 screening at all), the `if unclassified:` guard is True (unclassified list is non-empty) but `max_pool=all_element_ids` is empty → crash. In practice this is unlikely because elements screened from Phase 1 won't have unclassified ElementIdentification objects, but the guard is fragile.

**Fix:** Add a check `if not max_pool: return` at the start of `_gate_subset_relative_cl`.

---

### NEW-2 — DOCUMENTATION — LOW
**Title:** Line 1863 comment claims N_matched≥3 gate was "removed" but it still exists at line 4682

**Location:** `alias.py:1863` vs `alias.py:4682`

The comment at line 1863 states: "The previous CL>=adaptive_dt decision, the hard N_matched>=3 gate (the paper explicitly supports One-Line and Sparse-Line elements), and the Boltzmann-R^2 gate are all homegrown and were removed." This is factually wrong — the N_matched<3 CL-zeroing gate at line 4682 was NOT removed. The comment creates a false expectation for future maintainers, obscuring the actual behavior described in F2.

**Fix:** Update the comment at line 1863 to accurately state that the CL-zeroing gate at line 4682 still exists and explain why it is intentionally retained (or remove the gate if it truly should be gone).

---

## Verdict Table (Final)

| Finding | Real | Confirmed Severity |
|---------|------|--------------------|
| F1 | FALSE | N/A |
| F2 | TRUE | **HIGH** |
| F3 | TRUE | MEDIUM |
| F4 | TRUE | MEDIUM |
| F5 | TRUE | MEDIUM |
| F6 | FALSE | N/A |
| F7 | TRUE | LOW |
| F8 | TRUE | LOW |
| F9 | TRUE | MEDIUM |
| F10 | FALSE | N/A |
| NEW-1 | NEW | MEDIUM |
| NEW-2 | NEW | LOW |

**Highest confirmed severity: HIGH (F2 — CL zeroing for N_matched < 3 flows to detected=False via `_gate_relative_cl`)**
