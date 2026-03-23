---
phase: 01-physics-formalization
plan: 01
depth: full
one-liner: "Derived GPU-optimized Voigt profile (Weideman N=36, branch-free) and batched Boltzmann WLS (5-sum closed form) with complete dimensional verification and limiting cases"
subsystem: [derivation, formalism]
tags: [voigt-profile, faddeeva, boltzmann-plot, weighted-least-squares, JAX, GPU-parallelization, spectroscopy]

requires: []
provides:
  - "DERV-01: Voigt profile formulation via Faddeeva/Weideman with JAX parallelization strategy"
  - "DERV-02: Batched WLS Boltzmann fitting with covariance, outlier rejection, and Saha correction"
  - "Explicit FLOP counts and memory budgets for both kernels"
  - "Validation criteria for Phase 4 implementation testing"
affects: [01-02-PLAN, 03-jax-implementation, 04-validation]

methods:
  added: [Weideman-N36-rational-approximation, 2x2-WLS-normal-equations, pad-and-mask-batching]
  patterns: [broadcasting-outer-product-for-spectrum, 5-sum-reduction-for-WLS]

key-files:
  created:
    - ".gpd/research/derivations/derv01_voigt_profile.tex"
    - ".gpd/research/derivations/derv02_boltzmann_fitting.tex"

key-decisions:
  - "Weideman N=36 over Humlicek W4: branch-free enables JAX autodiff without NaN gradients"
  - "Normal equations over QR for 2x2 Boltzmann: condition number bounded by E_k spread (~10^2-10^3), 3x less overhead"
  - "Broadcasting over vmap for single-spectrum Voigt: maps to fused element-wise+reduction"
  - "Sigma-clipping over RANSAC for GPU outlier rejection: fully vectorizable, no random sampling"
  - "Float64 required for Voigt: coefficient dynamic range spans 14 orders of magnitude"

patterns-established:
  - "Convention assertion lines at top of every derivation .tex file"
  - "Inline dimension checks after every equation"
  - "Limiting case verification for each derived formula"

conventions:
  - "gamma = Lorentzian HWHM [nm], FWHM_L = 2*gamma"
  - "sigma_G = Gaussian standard deviation [nm], FWHM_G = 2.3548*sigma_G"
  - "z = (lambda - lambda_0 + i*gamma) / (sigma_G * sqrt(2)), dimensionless"
  - "V(lambda) = Re[w(z)] / (sigma_G * sqrt(2*pi)), [nm^-1]"
  - "y = ln(I*lambda/(g_k*A_ki)), x = E_k [eV], slope = -1/(k_B*T) [eV^-1]"
  - "k_B = 8.617333e-5 eV/K"

plan_contract_ref: ".gpd/phases/01-physics-formalization/01-01-PLAN.md#/contract"
contract_results:
  claims:
    claim-voigt-formalization:
      status: passed
      summary: "Derived complete Weideman N=36 Voigt formulation: branch-free polynomial eval, 236 FLOPs/point, <1e-13 relative error in float64. Three limiting cases verified analytically."
      linked_ids: [deliv-voigt-derivation, test-voigt-limits, test-voigt-dimensions, ref-weideman1994]
    claim-boltzmann-formalization:
      status: passed
      summary: "Derived closed-form 2-parameter WLS solvable via 5 dot products per batch element (10*N_max+7 FLOPs). Pad-and-mask batching eliminates Python loops. No vmap needed."
      linked_ids: [deliv-boltzmann-derivation, test-boltzmann-limits, test-boltzmann-dimensions, ref-tognoni2010]
  deliverables:
    deliv-voigt-derivation:
      status: passed
      path: ".gpd/research/derivations/derv01_voigt_profile.tex"
      summary: "6-section LaTeX derivation: physics motivation, Faddeeva algorithms (Weideman/Humlicek/Zaghloul comparison), error analysis, JAX parallelization strategy, 3 limiting cases, validation criteria"
      linked_ids: [claim-voigt-formalization, test-voigt-limits, test-voigt-dimensions]
    deliv-boltzmann-derivation:
      status: passed
      path: ".gpd/research/derivations/derv02_boltzmann_fitting.tex"
      summary: "7-section LaTeX derivation: physics motivation, closed-form WLS, covariance/uncertainty, batched GPU formulation with shapes, outlier rejection comparison, Saha correction, validation criteria"
      linked_ids: [claim-boltzmann-formalization, test-boltzmann-limits, test-boltzmann-dimensions]
  acceptance_tests:
    test-voigt-limits:
      status: passed
      summary: "All three limiting cases verified analytically in DERV-01 Sec. 5: (1) gamma->0 recovers Gaussian via w(x)=exp(-x^2) on real axis, (2) sigma->0 recovers Lorentzian via w(z)~i/(sqrt(pi)*z) asymptotic with sigma_G cancellation, (3) |z|->inf gives Lorentzian wings"
      linked_ids: [claim-voigt-formalization, deliv-voigt-derivation]
    test-voigt-dimensions:
      status: passed
      summary: "Dimensional chain verified: [z]=dimensionless, [w(z)]=dimensionless, [V(lambda)]=[nm^-1] (integrates to 1 over nm). All intermediate expressions checked inline."
      linked_ids: [claim-voigt-formalization, deliv-voigt-derivation]
    test-boltzmann-limits:
      status: passed
      summary: "All three limiting cases verified in DERV-02 Sec. 6: (1) w_i=1 reduces to OLS slope formula, (2) N=2 gives exact fit with zero residual, (3) all E_k equal gives D=0 detected as singular"
      linked_ids: [claim-boltzmann-formalization, deliv-boltzmann-derivation]
    test-boltzmann-dimensions:
      status: passed
      summary: "Dimensional chain verified: [y]=dimensionless, [x]=[eV], [b]=[eV^-1], [T_K]=1/([eV^-1]*[eV/K])=[K]. Covariance: [Cov(a,a)]=dimensionless, [Cov(b,b)]=[eV^-2]. All inline."
      linked_ids: [claim-boltzmann-formalization, deliv-boltzmann-derivation]
  references:
    ref-weideman1994:
      status: completed
      completed_actions: [cite, use]
      missing_actions: []
      summary: "Weideman (1994) SIAM J. Numer. Anal. 31(5) -- source of N=36 rational approximation, Moebius transform, coefficients, and error bounds (<1e-13 in float64)"
    ref-zaghloul2024:
      status: completed
      completed_actions: [cite]
      missing_actions: [compare]
      summary: "Zaghloul (2024) arXiv:2411.00917 -- cited as accuracy reference for Phase 4 numerical validation. Direct comparison deferred to implementation phase."
    ref-tognoni2010:
      status: completed
      completed_actions: [cite, use]
      missing_actions: []
      summary: "Tognoni et al. (2010) Spectrochim. Acta B 65 -- standard CF-LIBS methodology reference for Boltzmann plot fitting and closure"
    ref-exojax:
      status: completed
      completed_actions: [cite]
      missing_actions: [compare]
      summary: "Kawahara et al. (2022) ExoJAX -- cited as prior art for JAX-based Voigt computation. Direct benchmarking deferred to Phase 4."
    ref-olivero1977:
      status: completed
      completed_actions: [cite, use]
      missing_actions: []
      summary: "Olivero & Longbothum (1977) JQSRT 17 -- Voigt FWHM approximation formula (0.02% accuracy) included in DERV-01 for validation use"
  forbidden_proxies:
    fp-fabricated-benchmarks:
      status: rejected
      notes: "No simulated timing numbers or fabricated accuracy comparisons produced. FLOP counts and memory budgets derived from algorithm analysis, not simulation."
    fp-qualitative-speedup:
      status: rejected
      notes: "Explicit FLOP counts derived: 236 FLOPs/Voigt eval, 10*N_max+7 FLOPs/Boltzmann fit. Memory budgets stated with formulas. No qualitative claims."
  uncertainty_markers:
    weakest_anchors:
      - "Weideman N=36 accuracy in float32 for |z| > 20: documented as 0.1-1% wing errors, outside typical LIBS regime"
    unvalidated_assumptions:
      - "V100S float64 throughput of 7.8 TFLOPS assumes sustained compute (not memory-limited) [UNVERIFIED - training data]"
    competing_explanations: []
    disconfirming_observations: []

duration: 25min
completed: 2026-03-23
---

# Plan 01-01: Core Kernel Formalization Summary

**Derived GPU-optimized Voigt profile (Weideman N=36, branch-free) and batched Boltzmann WLS (5-sum closed form) with complete dimensional verification and limiting cases**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-03-23T17:54:00Z
- **Completed:** 2026-03-23T18:19:00Z
- **Tasks:** 2/2
- **Files created:** 2

## Key Results

- Weideman N=36 Voigt kernel: 236 real FLOPs per evaluation, branch-free, <1e-13 relative error in float64. Humlicek W4 ruled out for JAX due to NaN gradients from `jnp.where` evaluating all branches during AD. [CONFIDENCE: HIGH]
- Boltzmann WLS: closed-form via 5 dot products + 2 divisions per batch element (10*N_max + 7 FLOPs). Pad-and-mask batching on (B, N_max) arrays, no vmap needed. [CONFIDENCE: HIGH]
- Memory budgets: 400 MB per spectrum (10k wavelengths x 5k lines x 8 bytes), 2 MB for 1000-element Boltzmann batch (N_max=50). [CONFIDENCE: HIGH]

## Task Commits

1. **Task 1: Derive GPU-optimized Voigt profile (DERV-01)** - `9ffb4dc` (derive)
2. **Task 2: Formalize vectorized Boltzmann fitting (DERV-02)** - `9865e4c` (derive)

## Files Created

- `.gpd/research/derivations/derv01_voigt_profile.tex` -- Voigt profile from physics through Weideman algorithm to JAX spec
- `.gpd/research/derivations/derv02_boltzmann_fitting.tex` -- Boltzmann WLS from population statistics through closed-form to batched GPU

## Next Phase Readiness

- DERV-01 provides the mathematical specification for JAX Voigt kernel implementation (Phase 3)
- DERV-02 provides the batched WLS formulation for GPU Boltzmann fitting (Phase 3)
- Both derivations define quantitative validation criteria for Phase 4
- Convention lock confirmed: gamma=HWHM, sigma=std dev, wavelength-domain throughout

## Contract Coverage

- Claim IDs: claim-voigt-formalization -> passed, claim-boltzmann-formalization -> passed
- Deliverable IDs: deliv-voigt-derivation -> passed, deliv-boltzmann-derivation -> passed
- Acceptance test IDs: test-voigt-limits -> passed, test-voigt-dimensions -> passed, test-boltzmann-limits -> passed, test-boltzmann-dimensions -> passed
- Reference IDs: ref-weideman1994 -> cited/used, ref-zaghloul2024 -> cited, ref-tognoni2010 -> cited/used, ref-exojax -> cited, ref-olivero1977 -> cited/used
- Forbidden proxies: fp-fabricated-benchmarks -> rejected, fp-qualitative-speedup -> rejected

## Equations Derived

**Eq. (01-01.1):** Voigt profile via Faddeeva function

$$V(\lambda) = \frac{\operatorname{Re}[w(z)]}{\sigma_G\sqrt{2\pi}}, \quad z = \frac{(\lambda - \lambda_0) + i\gamma}{\sigma_G\sqrt{2}}$$

**Eq. (01-01.2):** Weideman N=36 approximation

$$w(z) \approx \frac{1}{\sqrt{\pi}}\frac{L}{L-iz} + \frac{2}{(L-iz)^2}\sum_{n=0}^{35} c_n Z^n, \quad Z = \frac{L+iz}{L-iz}, \quad L \approx 5.0454$$

**Eq. (01-01.3):** Boltzmann plot linear model

$$y = \ln\!\left(\frac{I\lambda}{g_k A_{ki}}\right) = -\frac{E_k}{k_B T} + a$$

**Eq. (01-01.4):** Closed-form WLS slope and intercept

$$b = \frac{S_w S_{wxy} - S_{wx} S_{wy}}{S_w S_{wxx} - S_{wx}^2}, \quad a = \frac{S_{wxx} S_{wy} - S_{wx} S_{wxy}}{S_w S_{wxx} - S_{wx}^2}$$

**Eq. (01-01.5):** Temperature from slope

$$T_K = -\frac{1}{b \cdot k_{B,\text{eV}}}, \quad \sigma_T = T_K^2\,k_{B,\text{eV}}\,\sigma_b$$

## Validations Completed

- **Voigt limiting cases (analytical):** gamma->0 recovers Gaussian, sigma->0 recovers Lorentzian (sigma_G cancels exactly), |z|->inf gives Lorentzian wings
- **Boltzmann limiting cases (analytical):** w_i=1 gives OLS, N=2 gives exact fit, all E_k equal gives det=0
- **Dimensional analysis:** All intermediate and final expressions checked. [V]=[nm^-1], [z]=dimensionless, [b]=[eV^-1], [T]=[K], [Cov(b,b)]=[eV^-2]
- **Convention consistency:** All width parameters use locked conventions (HWHM for Lorentzian, sigma for Gaussian). Verified against profiles.py and boltzmann.py.
- **Sign convention:** Boltzmann slope b < 0 for T > 0 (populations decrease with energy). Positive slope flagged as error.

## Decisions Made

- Weideman N=36 chosen over Humlicek W4 and Zaghloul 2024 based on branch-free property (critical for JAX AD) and existing codebase implementation
- Normal equations chosen over QR/SVD for 2x2 Boltzmann system (condition number bounded, 3x less overhead)
- Broadcasting approach preferred over vmap for single-spectrum Voigt (maps to fused kernel)
- Sigma-clipping recommended for GPU outlier rejection; RANSAC incompatible with JAX JIT; Huber reserved for refinement

## Deviations from Plan

None -- plan executed exactly as written.

## Approximations Used

| Approximation | Valid When | Error Estimate | Breaks Down At |
|---|---|---|---|
| Weideman N=36 (float64) | All z with Im(z) >= 0 | < 1e-13 relative | N/A in float64 |
| Weideman N=36 (float32) | \|z\| < 20 approximately | 0.1-1% in wings | \|z\| > 20 (coefficient truncation) |
| 2x2 normal equations | kappa(X^T W X) < 1e6 | ~2-3 digits lost | Highly correlated E_k values |
| Olivero-Longbothum FWHM | All sigma, gamma > 0 | 0.02% | N/A (empirical formula) |

## Issues Encountered

None.

## Open Questions

- Is mixed-precision (float16 for memory, float64 for compute) worth exploring for the Voigt outer product? Could reduce the 400 MB memory bottleneck.
- Should the batched Boltzmann include an automatic N_lines < 3 warning per group, or handle it silently with det=0 detection?

---

_Phase: 01-physics-formalization, Plan: 01_
_Completed: 2026-03-23_
