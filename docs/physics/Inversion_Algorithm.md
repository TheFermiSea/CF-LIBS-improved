# Physics: Iterative CF-LIBS Inversion Algorithm

This is a step-by-step walkthrough of the iterative CF-LIBS solver
(`cflibs.inversion.solver.IterativeCFLIBSSolver`). It is a companion to
[Equations](Equations.md), which states the equations, and
[Assumptions and Validity](Assumptions_And_Validity.md), which explains
when those equations are physically valid.

The reader is assumed to have seen at least one CF-LIBS paper (e.g.,
Tognoni et al. 2010 [2]; Hou et al. 2021 [10]). This page starts after
line identification has produced `LineObservation` objects; for the
preceding peak-detection and element-identification workflow, see
[Peak Identification and Line Matching](../user/Peak_Identification_Guide.md).

---

## Inputs and Outputs

### Inputs

- A list of `LineObservation` objects, each carrying `(wavelength_nm,
  intensity, intensity_error, element, ionization_stage, A_ki, g_k,
  E_k_ev)`. The line-detection step
  (`cflibs.inversion.line_detection.detect_line_observations`) produces
  these from a measured spectrum and an atomic database.
- Configuration: tolerances, closure mode, max iterations, pressure.
- Optional: initial `T`, initial `n_e`. Both are auto-estimated if not
  supplied.

### Outputs

A `CFLIBSResult`:

- `temperature_K`, `temperature_uncertainty_K`
- `electron_density_cm3`
- `concentrations: dict[str, float]` (mass fractions)
- `concentration_uncertainties: dict[str, float]`
- `boltzmann_fits: dict[str, BoltzmannFitResult]` (per element)
- `convergence_history: list[dict]`
- `iterations: int`, `converged: bool`
- `quality_metrics: QualityMetrics`

---

## Algorithm

```
Inputs : observations [LineObservation], config
Output : (T, n_e, {C_s}) with uncertainties

INIT:
  T  ← initial temperature estimate (default: from strongest element's
       Boltzmann plot using only its lines)
  n_e ← initial density estimate (default: 1e17 cm^-3)

REPEAT:
  ── Step 1: Saha correction ───────────────────────────────────────
  for each LineObservation L of element s, stage z:
      L.q_corrected ← L.intercept_term − z · ln S(T, n_e; s)
  end

  ── Step 2: Common-slope Boltzmann fit ────────────────────────────
  fit a single line through { (E_k, ln(I·λ/(g_k·A_ki)) + correction) }
  with all elements sharing the slope (= -1/(k_B T)) and per-element
  intercepts q_s. Method: SIGMA_CLIP / WEIGHTED_LS / RANSAC / HUBER.
  
  T_new ← -1 / (k_B · slope)

  ── Step 3: Closure equation ──────────────────────────────────────
  if mode == "standard":
      F     ← Σ_s U_s(T_new) · exp(q_s)
      C_s   ← U_s(T_new) · exp(q_s) / F
  elif mode == "matrix":
      F     ← U_m(T_new) · exp(q_m) / C_matrix      # m = matrix elt
      C_s   ← U_s(T_new) · exp(q_s) / F
  elif mode == "oxide":
      compute elemental C_s as in standard, then convert to oxide
      mass fractions using stoichiometry.

  ── Step 4: Update n_e ────────────────────────────────────────────
  Re-derive n_e from charge balance and the Saha equation evaluated
  at T_new and the inferred {C_s}. Either:
    (a) compute n_total = sum_s C_s · ρ / M_s  (if absolute density
        is known), then n_e = sum_s n_s · ⟨z_s⟩(T_new); or
    (b) hold buffer-gas pressure constant (default 101325 Pa) and
        adjust n_e so that ideal-gas + Saha is consistent.

  ── Convergence test ──────────────────────────────────────────────
  if  |T_new - T| / T   < t_tolerance  AND
      |n_e_new - n_e| / n_e < ne_tolerance:
      break

  T  ← T_new
  n_e ← n_e_new

UNTIL max_iterations

────────────────────────────────────────────────────────────────────
QUALITY METRICS:
  r_squared per element from Boltzmann fit
  inter-element T consistency
  neutral/ion T consistency (Saha-Boltzmann)
  McWhirter LTE check at converged (T, n_e)
  reduced chi-squared of forward-model fit (if reference spectrum
  available)

UNCERTAINTY (selectable):
  none        → no uncertainty propagation
  analytical  → Boltzmann fit covariance → propagated through closure
                via the `uncertainties` package (correlation-aware)
  monte_carlo → 200 re-runs of the entire pipeline with perturbed
                inputs (captures non-linearities)
  bayesian    → separate command (cflibs bayesian) — full posterior
```

---

## Where Each Step Lives in the Code

| Step | Module |
|------|--------|
| Peak detection and line matching | `cflibs.inversion.line_detection.detect_line_observations`; see [Peak Identification and Line Matching](../user/Peak_Identification_Guide.md) |
| Line scoring + selection | `cflibs.inversion.line_selection.LineSelector` |
| Saha correction | `IterativeCFLIBSSolver._saha_correction` |
| Common-slope Boltzmann fit | `cflibs.inversion.physics.boltzmann_fit` (and `boltzmann.py` shim) |
| Closure equation | `cflibs.inversion.physics.closure.ClosureEquation` |
| `n_e` update | `IterativeCFLIBSSolver._update_electron_density` |
| Quality assessment | `cflibs.inversion.quality.QualityAssessor` |
| Analytical UQ | `cflibs.inversion.uncertainty.create_boltzmann_uncertainties` |
| Monte Carlo UQ | `cflibs.inversion.uncertainty.MonteCarloUQ` |
| Bayesian | `cflibs.inversion.solve.bayesian` (separate solver) |

---

## Why Common-Slope Fitting (vs. Per-Element Fits)

A naive Boltzmann analysis fits each element separately, gets per-element
`T_s`, and averages them. With multiple elements this is statistically
worse than a single fit constrained to share `T` because:

- The slope estimate aggregates lever arms from all elements; the
  pooled `E_k` range is wider than any individual element's.
- Lines from different elements partially cancel each other's noise.
- The intercepts `q_s` factor cleanly into the closure equation
  without needing to reconcile element-disagreement on `T`.

The common-slope fit is the standard CF-LIBS approach since Ciucci 1999
[1]. It does, however, require that **all elements really do share the
same `T`** — i.e. the single-zone LTE assumption holds. If they do not,
the per-element residuals are large and the `temperature_consistency`
quality metric flags it.

---

## Why Iterate (vs. Solve in One Pass)

The Saha correction in step 1 needs `T` and `n_e`, which are what we are
solving for. An initial guess seeds the loop; subsequent passes refine
it. In practice:

- **Fast convergence** when LTE holds and the line set is clean: 3–6
  iterations.
- **Slow convergence** when self-absorption or non-LTE is corrupting
  data: hits `max_iterations` without true convergence. The
  `converged: false` flag in the result reports this.

If oscillation occurs (`T` ping-pongs without settling), the most common
cause is poor line selection — usually a self-absorbed strong line that
"wants" to set the slope. Rerun with `exclude_resonance: true` and a
tighter `min_snr`.

---

## How Uncertainties Propagate

The analytical UQ path is:

1. Each `LineObservation.intensity_error` becomes a weight `1/σ²` in
   the Boltzmann fit.
2. The fit returns a slope/intercept covariance matrix `Σ_q`.
3. `T_uncertainty = |∂T/∂slope| · σ_slope = (k_B T²) · σ_slope`.
4. For each element, `C_s = exp(q_s) U_s(T) / F` gives the
   propagation
   ```
   σ²_C_s = (∂C_s/∂q_s)² σ²_q_s
          + Σ_t≠s (∂C_s/∂q_t)² σ²_q_t
          + cross terms via Σ_q
          + (∂C_s/∂T)² σ²_T
   ```
   The `uncertainties` package handles correlation bookkeeping
   automatically; CF-LIBS just builds an `ufloat`-decorated version of
   the closure expression.

The Monte Carlo UQ path skips the analytical chain entirely:

1. Resample `intensity_i ∼ N(I_i, σ_i)` for every line, 200 times.
2. Run the full pipeline (detect → select → solve) on each sample.
3. Report mean and standard deviation of the resulting `(T, n_e, {C_s})`.

MC captures non-linearities and pipeline-stage interactions (e.g. a
noise sample causing a line to fall below `min_snr` and dropping out)
that analytical propagation cannot. Cost: ~200× a single solve.

The Bayesian path (`cflibs bayesian`) replaces the iterative algorithm
entirely with NumPyro NUTS sampling of the joint posterior — the gold
standard when you can afford it. See [Equations § Bayesian Forward
Model](Equations.md#bayesian-forward-model) for the likelihood and
priors used.

---

## See Also

- [Equations](Equations.md) — every equation invoked above.
- [Assumptions and Validity](Assumptions_And_Validity.md) — when each
  step is physically valid.
- [Quick Start: Real Data](../user/Quick_Start_Real_Data.md) — running
  the inversion from the CLI.
- [Peak Identification and Line Matching](../user/Peak_Identification_Guide.md) —
  how raw spectra become solver-ready `LineObservation` objects.
- [User Guide § Inversion Diagnostics](../user/User_Guide.md#inversion-diagnostics) —
  inspecting per-element Boltzmann fits, residuals, and convergence
  history.
- [Codebase Architecture](../reference/Codebase_Architecture.md) — full
  module map.
