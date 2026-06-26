# Adversarial Verification: cflibs/validation/round_trip.py Census Findings

**Verifier:** Adversarial sub-agent (claude-sonnet-4-6)
**Date:** 2026-06-25
**Method:** ripgrep + Read on actual source at worktree v4-m5; cross-referenced with
`saha-boltzmann-lte.md` and `cflibs-method.md` literature notes.

---

## Finding 1 — CRITICAL: Missing `hc/λ` in intensity formula

**Census claim:** `_append_stage_observations` uses `base_intensity * 1e-4` instead of the physical
`(hc/λ) * g*A*exp(-E/kT)/U`, causing a wavelength-dependent bias in recovered concentrations.

**REAL: TRUE** — confirmed.

**Corrected severity: CRITICAL** (upheld).

**Reasoning:** `round_trip.py:306` computes
`base_intensity = concentration * stage_fraction * g_k * A_ki * boltzmann_factor / U`
then `intensity = base_intensity * 1e-4` (line 309). The inversion's `LineObservation.y_value`
at `cflibs/inversion/common/data_structures.py:78` is
`ln(I * wavelength_nm / (g_k * A_ki))`. For the Boltzmann plot to self-cancel the wavelength
dependence in the y-axis, the forward intensity must include a `hc/λ` factor so that
`y = ln((hc/λ * g*A*exp(-E/kT)/U) * λ / (g*A)) = ln(hc * exp(-E/kT) / U)` — λ cancels.
With the `1e-4` approximation, `y = ln(1e-4 * λ * exp(-E/kT) / U)` and the `λ` term
persists in the intercept. The closure equation (`closure.py:721`) then uses `U_s * exp(q_s)`
to set relative concentrations: the `ln(λ_s)` shift in `q_s` is different for each element
because different species have transitions at different wavelengths, producing a
wavelength-dependent systematic bias in concentration ratios. This is a genuine physics error,
not a sign-convention difference or an already-handled edge case.

---

## Finding 2 — HIGH: `compute_equilibrium_ne` uses wrong denominator `(1 + avg_Z)`

**Census claim:** The denominator should be `(1 + 2*avg_Z)` because the total particle count
includes both heavy particles and electrons.

**REAL: FALSE** — the formula is correct; the variable name is misleading.

**Corrected severity: N/A** (false alarm — no bug).

**Reasoning:** Let N0 = heavy-particle number density (neutrals + all ions). By charge
neutrality, n_e = avg_Z * N0. The ideal-gas pressure is P = (N0 + n_e) * kT = N0 * (1 + avg_Z) * kT,
so N0 = P / (kT * (1 + avg_Z)). The code computes `n_total_m3 = P / (kT * (1 + avg_Z))` — this
IS N0 (heavy-particle density, misnamed as "n_total"), then `n_e_new = avg_Z * n_total_m3 = avg_Z * N0`,
which is the correct electron density. The `(1 + 2*avg_Z)` denominator would only apply if
`n_total_m3` represented all particles including electrons, but it doesn't — the math chain is
self-consistent. The docstring at line 457-459 says `n_total = P / (kT * (1 + Z_avg))` and
`n_e = Z_avg * n_total`, which describes the heavy-particle count correctly. The variable name
`n_total_m3` is misleading (it is really N0), but the calculation gives the right n_e.

---

## Finding 3 (original) — HIGH: S2 missing factor of 2 for electron spin degeneracy

**Census claim:** S2 formula omits the factor-of-2 electron spin degeneracy.

**REAL: FALSE** (census self-retracted in the original report, marked as non-finding).

**Corrected severity: N/A** — confirmed non-finding.

**Reasoning:** `constants.py:72-76` explicitly states: "The factor of 2 for electron spin
degeneracy (g_e = 2) is INCLUDED in this value." `SAHA_CONST_CM3 = 6.042e21` already includes
the factor of 2. Both S1 and S2 use `SAHA_CONST_CM3` identically, so neither is missing the
factor. The census correctly identified this as a false alarm during the audit process.

---

## Finding 3 (revised) — HIGH: `NoiseModel` shot noise applied to background-inflated signal; uncertainty formula circular

**Census claim:** Shot noise is computed from `intensity` after background has been added
(inflating Poisson variance). The uncertainty formula then uses the post-noise intensity,
creating a circular estimate.

**REAL: TRUE** — confirmed.

**Corrected severity: HIGH** (upheld).

**Reasoning:** `round_trip.py:588-615` confirms the sequence: (1) `intensity += self.background`
(line 591) adds background before shot noise; (2) `shot_std = np.sqrt(max(intensity, 1.0))`
(line 600) applies Poisson approximation to the signal+background sum, inflating the shot
variance by background counts; (3) `var_shot = max(intensity, 1.0)` (line 612) uses the
final noisy `intensity` (which already encodes the shot realization) as the variance estimate.
Physical Poisson shot noise should be applied to photon counts (the signal before background
subtraction), and the uncertainty estimate should be derived from known noise parameters
applied to the clean signal, not from the noise-contaminated measurement. This inflates the
uncertainty for bright background, and makes the uncertainty formula depend on the noise
realization rather than the noise model parameters — the "circular" diagnosis is accurate.

---

## Finding 4 — MEDIUM: Boltzmann factor uses `KB_EV * temperature_K` instead of pre-computed `T_eV`

**Census claim:** Clarity/consistency issue — `KB_EV * temperature_K` recomputes what `T_eV`
already is.

**REAL: TRUE** — confirmed as a clarity issue.

**Corrected severity: MEDIUM** (upheld, no math error).

**Reasoning:** `round_trip.py:305` is `np.exp(-E_k / (KB_EV * temperature_K))` while
`T_eV = temperature_K / EV_TO_K` is computed at line 205 and passed into the Saha computation
but not forwarded to `_append_stage_observations`. `KB_EV = 8.617e-5 eV/K` and
`EV_TO_K = 1 / KB_EV`, so `KB_EV * temperature_K == T_eV` algebraically — no numeric error.
The inconsistency invites future maintenance mistakes and should be fixed, but it is not a
physics bug.

---

## Finding 5 — HIGH: `IterativeCFLIBSSolver` hardwired; no solver injection seam

**Census claim:** `RoundTripValidator` imports and instantiates `IterativeCFLIBSSolver` via
deferred import at line 779; no way to inject other `SolverStrategy` implementations.

**REAL: TRUE** — confirmed.

**Corrected severity: HIGH** (upheld).

**Reasoning:** `round_trip.py:779-781` performs
`from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver; solver = IterativeCFLIBSSolver(self.atomic_db, **solver_kwargs)`.
The `RoundTripValidator.__init__` signature (lines 665-691) accepts no `solver_factory`
parameter. The `SolverStrategy` ABC exists in the codebase. This means the validator cannot
gate regressions in the Bayesian, joint L-BFGS-B, or closed-form ILR solvers — those require
separate bespoke tests or manual wiring. The deferred import pattern makes the dependency
invisible at the constructor call site. This is a real architectural gap, not merely
theoretical.

---

## Finding 10 — HIGH: Round-trip test asserts 50% temperature error tolerance

**Census claim:** `test_round_trip_no_noise` asserts `result.temperature_error_frac < 0.50`,
which is 10× looser than the validator default of 5%. `test_round_trip_with_noise` asserts
only `result.converged` with no accuracy check.

**REAL: TRUE** — confirmed.

**Corrected severity: HIGH** (upheld).

**Reasoning:** `tests/test_round_trip.py:258-260` reads:
`assert result.temperature_error_frac < 0.50`. The `RoundTripValidator` has a
`temperature_tolerance=0.10` (10%) in that test's constructor call (line 241), but the test
asserts 50%. Line 282 in `test_round_trip_with_noise` checks only `assert result.converged`
with no temperature or concentration accuracy check. A 40% temperature error or a completely
wrong concentration set would pass both tests silently. The `result.passed` attribute (which
uses the configured tolerances) is never asserted in any of these round-trip tests.

---

## Finding 11 — HIGH: NIST parity tolerances guard only against catastrophic failure

**Census claim:** Partition-function tolerances of 65–70% and ionization-fraction tolerances
up to 150% are too loose to catch real regressions that matter for concentration recovery.

**REAL: TRUE** — confirmed.

**Corrected severity: HIGH** (upheld).

**Reasoning:** `tests/test_nist_parity.py:110-115` sets `tol = 0.70` (T >= 15000 K) and
`tol = 0.65` (T <= 10000 K). Line 340 permits 150% relative error for minority ionization
stages (nist_f < 0.05). Per the literature note `saha-boltzmann-lte.md §P6`, partition-function
errors propagate linearly into concentration recovery via `C_s ∝ U_s(T) * exp(q_s)`. A 70%
error in U_s(T) translates directly into a 70% error in C_s — well outside any physically
meaningful recovery target. The 150% tolerance on minority stages is particularly problematic
because Saha correction for ionic lines uses those minor-stage fractions. The note in the test
code attributes the large discrepancy to energy-level cleanup and autoionizing levels, but the
tolerance should be tightened with documented acknowledged deviations tracked to specific issues
rather than absorbed into a blanket 70% / 150% pass window.

---

## Summary of Verdicts

| # | Census Severity | REAL? | Confirmed Severity |
|---|-----------------|-------|--------------------|
| 1 | CRITICAL | TRUE | CRITICAL |
| 2 | HIGH | FALSE | N/A (no bug) |
| 3 (original) | HIGH | FALSE | N/A (self-retracted) |
| 3 (revised) | HIGH | TRUE | HIGH |
| 4 | MEDIUM | TRUE | MEDIUM |
| 5 | HIGH | TRUE | HIGH |
| 6 | MEDIUM | TRUE (design dup) | MEDIUM |
| 7 | MEDIUM | TRUE | MEDIUM |
| 8 | MEDIUM | TRUE | MEDIUM |
| 9 | LOW | TRUE | LOW |
| 10 | HIGH | TRUE | HIGH |
| 11 | HIGH | TRUE | HIGH |

**Highest confirmed severity: CRITICAL** (Finding 1 — missing `hc/λ`).

---

## New Findings (Missed by Census)

### New Finding A — MEDIUM: Silent fallback to physically meaningless synthetic transitions

**Location:** `round_trip.py:364-391`

**Evidence:** `_generate_synthetic_transitions` generates `E_k_ev = rng.uniform(0.5, 6.0)`,
`g_k = rng.integers(3, 15)`, `A_ki = 10**rng.uniform(6, 8)` from a deterministic seed. If
the atomic database is unavailable or returns no transitions for an element/stage pair
(including a common DB-unavailable scenario in CI), the round-trip test silently proceeds with
these fictitious atomic data. The recovered temperature and concentrations are then compared
to ground truth but were generated with fake atomic parameters that have no correlation to
real LIBS physics. The test passes if the inversion happens to converge, even though the
data are physically meaningless. **Fix:** add a warning log AND a `metadata["used_synthetic_fallback"]`
flag that the test can assert is `False` to ensure DB-backed transitions were used.

### New Finding B — LOW: `NoiseModel` clamps negative intensity to `1.0`, not `0.0`

**Location:** `round_trip.py:608`

**Evidence:** `intensity = max(intensity, 1.0)` forces a minimum of 1.0 on all noisy
intensities, even those that physically should be undetected (i.e., signal < 0 after
noise). Clamping to 1.0 rather than 0.0 means all "dark" lines have a fabricated positive
intensity equal to the minimum detection unit. This is noted in the census as Finding 8
(the "phantom-line bias"), but the more precise observation is that 1.0 is arbitrary in the
same undocumented unit system as the `1e-4` scale. If Finding 1 is fixed and intensities
are in physical units, the clamp floor of `1.0` will be either catastrophically high or
negligibly low — it will need to be updated in the same pass. This cross-dependency is not
noted in the census.
