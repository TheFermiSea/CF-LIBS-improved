# cflibs/manifold — Adversarial Verification of Census Findings

Verifier: claude-sonnet-4-6 (adversarial pass)
Date: 2026-06-25
Worktree: `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5`
Census source: `scratchpad/overhaul/census/manifold.md`

---

## Verdict Summary

| ID  | Title (census)                                      | REAL?  | Corrected Severity | Notes |
|-----|-----------------------------------------------------|--------|--------------------|-------|
| F1  | Total Heavy-Particle Density Approx as n_e          | TRUE   | **medium** (↓ from high) | Confirmed code; standard CF-LIBS approximation, not an error |
| F2  | Gate Delay Configured but Never Applied             | TRUE   | **high**           | Confirmed; gate_delay_s never passed to linspace |
| F3  | Stark Temperature Scaling Missing in batch_forward  | TRUE   | **medium**         | Confirmed; no T^(-alpha) in batch_forward.py:447 |
| F4  | BasisLibraryGenerator Emissivity Drops hc/4π       | TRUE   | **low**            | Confirmed benign; area normalization cancels it |
| F5  | Three Parallel Physics Implementations              | TRUE   | **medium** (↓ from high) | Confirmed duplication; F3 is consequence; but each path serves different API consumers |
| F6  | vector_index.py Imports from cflibs.inversion.common.pca | TRUE | **low**       | Confirmed; reverse dependency exists |
| F7  | gate_delay_s Stored in Config but Not in HDF5 Attrs | TRUE  | **low**            | Confirmed; also broader — gate_width_s, cooling params absent too |
| F8  | Triple Python Loop in _build_param_grid             | TRUE   | **low** (↓ from medium) | Confirmed; but grid sizes in practice are small |
| F9  | physics_version/use_voigt/use_stark Vestigial       | TRUE   | **medium**         | Confirmed; written to attrs only, no dispatch effect |
| F10 | LDM Path Has No Integration-Level Parity Test      | TRUE   | **high**           | Confirmed; no parity test, no gate_delay test |

**Highest confirmed severity: high** (F2, F10)

---

## Detailed Verdicts

### F1 — Total Heavy-Particle Density Approximated as n_e
**REAL: TRUE** | **Corrected severity: medium (was high)**

Code verified at `generator.py:567`: `n_species_total = element_conc * n_e` and `batch_forward.py:427`: `n_k = C_line * n_e * f_line * boltz`. Both confirmed. The census claims this is "wrong in principle" but the standard CF-LIBS literature (Ciucci 1999, Tognoni 2010 — cross-checked against `literature/cflibs-method.md`) treats this as a canonical approximation: for a predominantly singly-ionized plasma, `N_total ≈ n_e` holds because charge neutrality gives `n_e = Σ_s C_s z_s N_total`, and for z_eff~1 (typical LIBS conditions 0.7–1.0 eV) the error is small. The census itself acknowledges "standard CF-LIBS approx" and "≤20% at typical LIBS conditions." The approximation becomes 20–50% in error only above 1.2 eV where Fe/Ca become doubly ionized — this is the hot end of the manifold grid, not the typical operating regime. The severity should be medium rather than high because (a) it is the canonical CF-LIBS practice, not an idiosyncratic implementation error; (b) the error is bounded and regime-dependent; (c) the manifold is used for pattern matching / nearest-neighbor retrieval where relative shape matters more than absolute scale; (d) both code paths are self-consistent with each other.

### F2 — Gate Delay Configured but Never Applied
**REAL: TRUE** | **Severity: high (confirmed)**

Verified at `generator.py:986`: `times = jnp.linspace(0, gate_width_s, time_steps)` and `generator.py:1055`: `times = jnp.linspace(0, gate_width_s, time_steps)`. Both `_time_integrated_spectrum_ldm` and `_time_integrated_spectrum` start the time grid at t=0 rather than t=`gate_delay_s`. The `gate_delay_s` field is declared in `config.py:107` and loaded from YAML at `config.py:184`, but it is never passed to either time-integration function. Confirmed by checking the call sites at `generator.py:1365–1375` and `generator.py:1389–1398` — neither passes `self.config.gate_delay_s`. Physically, the gate delay is specifically designed to exclude the early optically-thick, high-temperature phase; starting integration at t=0 defeats this design intent. This is a genuine physics bug, not a documentation issue.

### F3 — Stark Temperature Scaling Missing in batch_forward.py
**REAL: TRUE** | **Severity: medium (confirmed)**

Verified at `batch_forward.py:447`: `gamma_S = 0.5 * line_stark * (n_e / _STARK_REF_NE)` — no `T^(-alpha)` factor. Confirmed against `generator.py:659`: `factor_T = jnp.power(jnp.maximum(T_eV, 0.1) / REF_T_EV, -l_stark_alpha)` with `return 0.5 * w_ref * factor_ne * factor_T`. The temperature correction is present in the main generator but absent in `batch_forward.py`'s `single_spectrum_forward` and `single_spectrum_forward_ldm`. Over the 0.5–2.0 eV manifold T range, the correction is `(0.5/0.862)^0.5 = 0.76` to `(2.0/0.862)^0.5 = 1.52`, i.e., up to 50% Stark width error. Severity kept at medium because (a) Stark broadening is a secondary effect for line intensity integrals; (b) the `batch_forward.py` path is a distinct API consumer from the main manifold generator; but it should be consistent.

### F4 — BasisLibraryGenerator Emissivity Drops hc/4π Factor
**REAL: TRUE** | **Severity: low (confirmed, census assessment correct)**

Verified at `basis_library.py:306`: `eps = trans.A_ki * n_k / trans.wavelength_nm` vs `generator.py:812`: `epsilon = (H_PLANCK * C_LIGHT / (4 * jnp.pi * l_wl * 1e-9)) * l_aki * n_upper_m3`. The `hc/(4π)` factor is absent. However, area normalization at `basis_library.py:330–332` (`spectrum /= area`) cancels any multiplicative scale factor. The `1/λ_nm` vs `1/λ_m` discrepancy (factor of 1e9) is similarly irrelevant after area normalization. The census assessment is correct: benign for NNLS decomposition context, but the docstring should document this to prevent misuse. Severity: low.

### F5 — Three Parallel Physics Implementations of Saha-Boltzmann-Voigt
**REAL: TRUE** | **Corrected severity: medium (was high)**

Confirmed: `generator.py` (static `@jit` kernels), `batch_forward.py` (`single_spectrum_forward` and `single_spectrum_forward_ldm`), and `basis_library.py` (`_compute_element_spectra`) are three distinct implementations of the same physics. F3 is a concrete proof of divergence. However, the original high severity overstates the risk: `basis_library.py` serves a fundamentally different purpose (CPU sequential, area-normalized, NNLS-oriented) and the `batch_forward.py` T1-2 adapter path (`forward_from_snapshot`) already delegates to `cflibs.radiation.kernels.forward_model`. The duplication is a maintenance liability rather than an active correctness bug. Severity downgraded to medium.

### F6 — vector_index.py Imports from cflibs.inversion.common.pca
**REAL: TRUE** | **Severity: low (confirmed)**

Verified at `vector_index.py:13`: `from cflibs.inversion.common.pca import PCAPipeline`. This creates a reverse dependency (manifold → inversion). `PCAPipeline` is a generic dimensionality-reduction utility with no inversion-specific state. Low severity because it's a design smell rather than a correctness bug and there's no circular import risk (inversion imports from manifold, not pca).

### F7 — gate_delay_s Stored in Config and HDF5 Attrs but Silently Unused
**REAL: TRUE** | **Severity: low (confirmed, and broadened)**

Verified at `generator.py:1415–1422`: the attrs block writes `elements`, `wavelength_range`, `temperature_range`, `density_range`, `physics_version`, `use_voigt_profile`, `use_stark_broadening`, `instrument_fwhm_nm`. `gate_delay_s` is not written. Additionally, `gate_width_s`, `cooling_t0_s`, `cooling_temperature_exponent`, and `cooling_density_exponent` are also not stored — the census finding is correct but incomplete. A manifold consumer cannot reconstruct the time-integration model from the stored file. This compounds F2.

### F8 — Triple Python Loop in _build_param_grid Pre-Allocates Large List
**REAL: TRUE** | **Corrected severity: low (was medium)**

Verified at `generator.py:1168–1174`: three nested Python for-loops with `.tolist()`. The census fix (numpy meshgrid + column_stack) would indeed be faster for large grids. However, this runs once per manifold generation (not per spectrum), and practical manifold grids (50 T × 20 n_e × 20 comp = 20,000 rows) complete in milliseconds. The "~2s for 1M rows" claim is correct but 1M-row grids are edge cases. Downgraded to low because this is a non-urgent performance polish item on a one-shot setup step.

### F9 — physics_version, use_voigt_profile, use_stark_broadening Are Vestigial
**REAL: TRUE** | **Severity: medium (confirmed)**

Verified at `generator.py:1419–1421`: these three fields are only written to HDF5 attrs, never read for dispatch in `generator.py`. `broadening_mode` (the actual dispatch flag via `BroadeningMode` enum) is not written to attrs at all. A user setting `use_stark_broadening: false` would observe no change in output — a silent misdirection. This is a real documentation/correctness issue. `physics_version` is additionally meaningless as there's no version-dispatch code.

### F10 — LDM Path in Generator Has No Integration-Level Parity Test
**REAL: TRUE** | **Severity: high (confirmed)**

Verified: `tests/manifold/` contains `test_emissivity_absolute_scale.py`, `test_generator_physics_w3.py`, and `test_sampling_guard.py`. There is no `test_ldm_parity.py`. `test_generator_physics_w3.py` contains a `test_ldm_sigma_grid_endpoint_uses_maxwell_std` test for the sigma grid but NO parity check between LDM output and Voigt output. No gate_delay test exists anywhere in the manifold test suite. A misconfigured `sigma_grid` or a regression in `ldm_broaden` could produce silent zeros with no failing test. High severity confirmed.

---

## Missed Findings (Spotted During Verification)

### MF1 — Multiple Cooling/Time-Model Parameters Not Stored in HDF5 Attrs
**Severity: low**

The census F7 notes `gate_delay_s` is not stored in HDF5 attrs. Verification revealed that `gate_width_s`, `cooling_t0_s`, `cooling_temperature_exponent`, and `cooling_density_exponent` are also absent from the attrs block (`generator.py:1414–1422`). A manifold loaded from disk is uninterpretable without these: there is no way to reconstruct what time-integration model generated a stored manifold, or to validate a loaded manifold is consistent with the configured parameters. The fix is to add all time-integration parameters to the attrs block alongside `gate_delay_s` once F2 is fixed.

### MF2 — broadening_mode Not Stored in HDF5 Attrs Despite Being the Active Dispatch Flag
**Severity: low**

`broadening_mode` (`BroadeningMode` enum) is the actual dispatch flag controlling whether the manifold was generated with per-line Voigt or LDM-Gaussian broadening. It is not written to attrs. A consumer loading a manifold cannot determine which broadening model was used — Voigt and LDM outputs have different spectral shapes (Lorentzian wings vs. Gaussian-only). Combined with F9 (where the vestigial `use_voigt_profile` IS stored but has no meaning), the stored attrs are actively misleading. Fix: write `attrs["broadening_mode"] = self.config.broadening_mode.value` alongside the other attrs.

---

## Verification Notes on "Already Handled" Concerns

Per the task brief, two prior false findings occurred in this codebase: (1) a "dead" class that was actually wired in, and (2) a Saha ionization-potential guard flagged wrong that was actually correct. No analogous false positives were found in this manifold audit:

- The `_calculate_saha_fractions` code correctly uses the two-stage Saha equation with IPD (not just a raw-IP guess).
- The SAHA_CONST_CM3 = 6.042e21 is numerically correct (computed: 6.037e21; difference < 0.1% from rounding).
- The `n_upper * 1e6` cm^-3 → m^-3 unit conversion at `generator.py:808` is correctly applied.
- The 0.5× factor (HWHM from FWHM) for Stark broadening is correctly applied in all paths.
- The `broadening_mode` enum dispatch in `generate_manifold` is real and active — it selects between the LDM path and the Voigt path.
- `gate_delay_s` truly has zero callers in the generation code paths (confirmed by exhaustive grep).
