---
phase: 01-physics-formalization
plan: 02
depth: full
one-liner: "Derived Anderson-accelerated Saha-Boltzmann solver and softmax/ILR closure parameterizations with complete Jacobians, convergence analysis, and limiting-case verification"
subsystem: [derivation, formalism]
tags: [anderson-acceleration, fixed-point, softmax, ILR, compositional-data, saha-boltzmann, JAX]

requires:
  - phase: 01-physics-formalization (plan 01)
    provides: convention lock (T_eV, n_e cm^-3, SAHA_CONST_CM3)
provides:
  - "DERV-03: Anderson acceleration algorithm for Saha-Boltzmann charge-balance fixed-point"
  - "DERV-04: Softmax and ILR parameterizations for compositional closure with Jacobians"
  - "JAX lax.scan specification for Anderson iteration"
  - "Implicit differentiation formula via jax.custom_root"
  - "Recommendation: softmax primary (GPU), ILR diagnostic (Aitchison geometry)"
affects: [03-implementation, 04-benchmarking, paper-methods-section]

methods:
  added: [Anderson acceleration, Tikhonov-regularized LS, softmax reparameterization, ILR/Helmert transform]
  patterns: [fixed-point formulation of charge neutrality, circular buffer history for AA, implicit differentiation through fixed-point]

key-files:
  created:
    - .gpd/research/derivations/derv03_anderson_acceleration.tex
    - .gpd/research/derivations/derv04_softmax_closure.tex

key-decisions:
  - "Softmax recommended over ILR for primary GPU pipeline (no boundary singularity, simpler, already in codebase)"
  - "Anderson depth m=3-5 recommended; m=0 recovers Picard as fallback"
  - "Implicit differentiation via jax.custom_root recommended over differentiating through iterations"

patterns-established:
  - "Convention assertion at top of every derivation file"
  - "Inline dimension checks after every equation"
  - "Self-critique checkpoints every 3-4 derivation steps"

conventions:
  - "T_eV for temperature, n_e in cm^-3"
  - "SAHA_CONST_CM3 = 4.829e15 cm^-3 K^-3/2"
  - "Compositions C_i dimensionless number fractions, sum to 1"
  - "All theta, z, Jacobian entries dimensionless for closure"

plan_contract_ref: ".gpd/phases/01-physics-formalization/01-02-PLAN.md#/contract"
contract_results:
  claims:
    claim-anderson-formalization:
      status: passed
      summary: "Derived Anderson acceleration in both constrained and unconstrained forms, with convergence theorem (Toth-Kelley r-linear, Walker-Ni GMRES(m) superlinear), safeguarding strategy, and three verified limiting cases"
      linked_ids: [deliv-anderson-derivation, test-anderson-limits, test-anderson-dimensions, ref-walker-ni-2011, ref-evans2018]
    claim-softmax-formalization:
      status: passed
      summary: "Derived softmax and ILR with explicit Jacobians, condition number analysis (kappa=1 uniform, ~Cmax/Cmin boundary for softmax, log-singular for ILR), gradient flow comparison, and four verified limiting cases; softmax recommended for GPU pipeline"
      linked_ids: [deliv-softmax-derivation, test-softmax-limits, test-softmax-dimensions, ref-egozcue2003]
  deliverables:
    deliv-anderson-derivation:
      status: passed
      path: ".gpd/research/derivations/derv03_anderson_acceleration.tex"
      summary: "Complete LaTeX derivation with physics motivation, both AA formulations, convergence analysis, JAX spec, safeguarding, and limiting cases"
      linked_ids: [claim-anderson-formalization, test-anderson-limits, test-anderson-dimensions]
    deliv-softmax-derivation:
      status: passed
      path: ".gpd/research/derivations/derv04_softmax_closure.tex"
      summary: "Complete LaTeX derivation comparing softmax and ILR with Jacobians, condition numbers, gradient analysis, and limiting cases"
      linked_ids: [claim-softmax-formalization, test-softmax-limits, test-softmax-dimensions]
  acceptance_tests:
    test-anderson-limits:
      status: passed
      summary: "All three limiting cases verified analytically: (1) m=0 recovers Picard, (2) m=1 reduces to Aitken/secant via gamma=r_k/(r_k-r_{k-1}), (3) single element yields analytical quadratic"
      linked_ids: [claim-anderson-formalization, deliv-anderson-derivation]
    test-anderson-dimensions:
      status: passed
      summary: "Dimensional consistency verified at every equation: [r_k]=[n_e]=cm^-3, [gamma]=dimensionless, [Delta_R]=cm^-3, [Tikhonov lambda]=cm^-6"
      linked_ids: [claim-anderson-formalization, deliv-anderson-derivation]
    test-softmax-limits:
      status: passed
      summary: "All four limiting cases verified: (1) D=1 gives C_1=1, (2) D=2 gives sigmoid with sqrt(2) ILR scaling, (3) uniform theta gives C_i=1/D, (4) ILR round-trip identity proven via VV^T projector"
      linked_ids: [claim-softmax-formalization, deliv-softmax-derivation]
    test-softmax-dimensions:
      status: passed
      summary: "All quantities confirmed dimensionless: theta_i, C_i, J_ij, z_k, Helmert V_ij"
      linked_ids: [claim-softmax-formalization, deliv-softmax-derivation]
  references:
    ref-walker-ni-2011:
      status: completed
      completed_actions: [cite, use]
      missing_actions: []
      summary: "Walker & Ni (2011) Eq. 2.2 used as the unconstrained AA formulation; cited throughout Section 2-3"
    ref-evans2018:
      status: completed
      completed_actions: [cite]
      missing_actions: []
      summary: "Evans et al. (2018) cited for convergence rate improvement proof"
    ref-toth-kelley-2015:
      status: completed
      completed_actions: [cite, use]
      missing_actions: []
      summary: "Toth & Kelley convergence theorem stated and applied to Saha-Boltzmann contractivity analysis"
    ref-egozcue2003:
      status: completed
      completed_actions: [cite, use]
      missing_actions: []
      summary: "Egozcue et al. (2003) Helmert basis construction used for ILR derivation"
    ref-pollock-rebholz-2019:
      status: completed
      completed_actions: [cite]
      missing_actions: []
      summary: "Pollock & Rebholz cited for non-contractive operator convergence theory"
  forbidden_proxies:
    fp-fabricated-convergence:
      status: rejected
      notes: "No fabricated convergence curves or iteration counts; validation criteria stated for Phase 4"
    fp-qualitative-comparison:
      status: rejected
      notes: "Both Jacobians derived explicitly; condition numbers analyzed quantitatively"
  uncertainty_markers:
    weakest_anchors:
      - "No published AA applied to Saha-Boltzmann; novel application of established theory"
      - "Gradient vanishing for extreme compositions not quantified for typical LIBS ranges"
    unvalidated_assumptions:
      - "Picard contraction rate |g'| ~ 0.3-0.7 estimated but not measured"
      - "AA depth m=3 sufficient — needs Phase 4 numerical confirmation"
    competing_explanations: []
    disconfirming_observations: []

duration: 25min
completed: 2026-03-23
---

# Plan 01-02: Iterative Solver & Closure Formalization Summary

**Derived Anderson-accelerated Saha-Boltzmann solver and softmax/ILR closure parameterizations with complete Jacobians, convergence analysis, and limiting-case verification**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-03-23T17:54:07Z
- **Completed:** 2026-03-23T18:19:00Z
- **Tasks:** 2/2
- **Files modified:** 2

## Key Results

- Anderson acceleration update: $n_e^{(k+1)} = g(n_e^{(k)}) - (\Delta G_k + \Delta R_k)\gamma_k^*$ with $\gamma$ from regularized LS [CONFIDENCE: HIGH]
- Softmax Jacobian: $J_{ij} = C_i(\delta_{ij} - C_j)$, rank $D-1$, condition number $\kappa = 1$ at uniform [CONFIDENCE: HIGH]
- ILR Jacobian: $J_\mathrm{ilr} = J_\mathrm{sm} V$ where $V$ is Helmert basis [CONFIDENCE: HIGH]
- Recommendation: softmax for GPU pipeline (no boundary singularity), ILR for diagnostic [CONFIDENCE: MEDIUM — needs Phase 4 numerical confirmation of conditioning claims]

## Task Commits

1. **Task 1: Anderson acceleration for Saha-Boltzmann (DERV-03)** — `02944ce` (derive)
2. **Task 2: Softmax vs ILR closure (DERV-04)** — `4254634` (derive)

## Files Created/Modified

- `.gpd/research/derivations/derv03_anderson_acceleration.tex` — Full Anderson acceleration derivation with convergence analysis, JAX spec, safeguarding
- `.gpd/research/derivations/derv04_softmax_closure.tex` — Softmax/ILR comparison with Jacobians, condition numbers, gradient flow

## Next Phase Readiness

- AA algorithm fully specified for implementation in JAX `lax.scan`
- Softmax closure ready for integration with joint optimizer
- Implicit differentiation via `jax.custom_root` recommended for wrapping solver in gradient-based optimization
- Validation criteria defined for Phase 4 numerical testing

## Contract Coverage

- Claim IDs advanced: claim-anderson-formalization -> passed, claim-softmax-formalization -> passed
- Deliverable IDs produced: deliv-anderson-derivation -> passed, deliv-softmax-derivation -> passed
- Acceptance test IDs run: test-anderson-limits -> passed, test-anderson-dimensions -> passed, test-softmax-limits -> passed, test-softmax-dimensions -> passed
- Reference IDs surfaced: ref-walker-ni-2011 (cite, use), ref-evans2018 (cite), ref-toth-kelley-2015 (cite, use), ref-egozcue2003 (cite, use), ref-pollock-rebholz-2019 (cite)
- Forbidden proxies rejected: fp-fabricated-convergence, fp-qualitative-comparison

## Equations Derived

**Eq. (01-02.1): Anderson acceleration update (unconstrained)**

$$n_e^{(k+1)} = g(n_e^{(k)}) - (\Delta G_k + \Delta R_k)\,\gamma_k^*$$

where $\gamma_k^* = \arg\min_\gamma \|r_k - \Delta R_k\,\gamma\|^2 + \lambda\|\gamma\|^2$.

**Eq. (01-02.2): Softmax Jacobian**

$$J_{ij}^\mathrm{sm} = C_i(\delta_{ij} - C_j) = [\mathrm{diag}(C) - CC^T]_{ij}$$

**Eq. (01-02.3): ILR Jacobian**

$$J_\mathrm{ilr} = (\mathrm{diag}(C) - CC^T)\,V$$

**Eq. (01-02.4): Implicit differentiation through fixed point**

$$\frac{dn_e^*}{d\theta} = (I - g'(n_e^*))^{-1}\,\frac{\partial g}{\partial\theta}\bigg|_{n_e^*}$$

**Eq. (01-02.5): Softmax gradient (centered form)**

$$\frac{\partial L}{\partial\theta_i} = C_i\left(\frac{\partial L}{\partial C_i} - \sum_j C_j\frac{\partial L}{\partial C_j}\right)$$

## Validations Completed

- **Anderson m=0 limit:** Reduces to Picard $n_e^{(k+1)} = g(n_e^{(k)})$ (empty gamma vector)
- **Anderson m=1 limit:** Reduces to Aitken/secant acceleration $\gamma = r_k/(r_k - r_{k-1})$
- **Anderson single-element:** Analytical quadratic solution exists; AA unnecessary
- **Softmax D=1:** $C_1 = 1$ identically
- **Softmax D=2:** $C_1 = \sigma(\theta_1 - \theta_2)$ (logistic sigmoid)
- **Uniform theta:** $C_i = 1/D$ for both softmax and ILR
- **ILR round-trip:** $\mathrm{ilr\_inverse}(\mathrm{ilr}(C)) = C$ proven via projector identity
- **Dimensional analysis:** All equations checked inline (cm^-3 for AA quantities, dimensionless for closure)

## Decisions Made

- Used unconstrained AA formulation (Walker & Ni Eq. 2.2) as primary — more common in practice than constrained form
- Recommended `jax.custom_root` for implicit differentiation rather than differentiating through iterations
- Recommended softmax over ILR for GPU pipeline — quantitative comparison supports this

## Deviations from Plan

None — plan executed as specified.

## Approximations Used

| Approximation | Valid When | Error Estimate | Breaks Down At |
|---|---|---|---|
| Anderson depth m=3-5 | Contractive map, $\|g'\| < 1$ | Superlinear convergence rate | Non-contractive regime ($T > 5$ eV, heavy double ionization) |
| Softmax log-sum-exp | All real theta | Exact (no approximation) | $\|\theta_i\| > 700$ (float64 overflow) |
| Picard contraction rate 0.3-0.7 | Typical LIBS: $T \sim 1$ eV, $n_e \sim 10^{17}$ | Estimated from Saha $S \propto 1/n_e$ | High-T multiply-ionized regime |

## Issues Encountered

None.

## Open Questions

- What is the actual contraction rate $|g'(n_e^*)|$ across the LIBS parameter space? (Phase 4 measurement)
- Is Tikhonov regularization $\lambda = 10^{-6}\|\Delta R\|_F^2$ optimal, or should it be adaptive?
- For $D > 10$ elements, does the softmax condition number become problematic in practice?

---

_Phase: 01-physics-formalization, Plan: 02_
_Completed: 2026-03-23_
