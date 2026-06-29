# PRD: Physics And Algorithmic Fidelity Overhaul

## Context

This PRD captures a physics-first review of the CF-LIBS codebase and defines a
hierarchical remediation program focused on plasma thermodynamics, radiative
modeling, inversion correctness, identifiability, and numerical performance.

The review intentionally excludes codebase architecture concerns. The goal is to
improve physical fidelity, remove inconsistent state semantics, strengthen
validation, and recover computational efficiency without masking physically
meaningful observables.

## Review Summary

The current implementation is productive and broad in scope, but several core
paths mix physically different quantities or normalize away signals that should
be available to the inference process.

Observed themes:

1. Plasma-state semantics are inconsistent across modules.
   - `cflibs/plasma/saha_boltzmann.py`
   - `cflibs/inversion/solver.py`
   - `cflibs/inversion/bayesian.py`
   - `cflibs/manifold/generator.py`

2. The line-population and transition bookkeeping is too coarse for a
   spectroscopic code path.
   - `cflibs/plasma/saha_boltzmann.py`
   - `cflibs/atomic/database.py`
   - `cflibs/atomic/structures.py`

3. Forward-model fidelity differs substantially between the classic spectrum
   model, the Bayesian forward model, and the manifold generator.
   - `cflibs/radiation/spectrum_model.py`
   - `cflibs/inversion/bayesian.py`
   - `cflibs/manifold/generator.py`

4. Retrieval and embedding paths normalize away absolute intensity information
   that is physically informative for density, path length, throughput, and
   self-absorption.
   - `cflibs/manifold/loader.py`
   - `cflibs/manifold/vector_index.py`
   - `cflibs/inversion/hybrid.py`

5. Validation is mostly qualitative or monotonic.
   - `tests/test_physics_invariants.py`
   - `tests/test_plasma.py`
   - `tests/test_radiation.py`
   - `tests/test_bayesian.py`
   - `tests/test_manifold_physics.py`

## Primary Findings

### F1. Plasma composition, heavy-particle density, and electron density are not
treated as distinct physical quantities

The forward-model implementations use element concentrations in places where a
heavy-particle density scale should exist independently of `n_e`. This couples
composition to electron density and makes abundances partially responsible for
setting emissivity scale in a way that is not physically transparent.

### F2. The production Saha-Boltzmann implementation is only partially coupled

The plasma solver uses a sequential neutral -> ion -> doubly ionized update and
accepts `n_e` as an external input rather than solving a fully coupled
ionization-plus-charge-neutrality system. This is acceptable as a temporary
approximation but should not remain the canonical implementation for a physics
library.

### F3. Spectroscopic state identity is underspecified

Level populations are keyed by `(element, stage, energy)`. The database preserves
additional distinctions such as statistical weight and line-specific metadata.
This creates a risk of aliasing distinct upper states when energies coincide or
are rounded together.

### F4. Forward-model fidelity is inconsistent across implementations

The classic spectrum model uses a legacy scalar broadening mode and transition
prefiltering by relative intensity, while Bayesian and manifold paths render
Voigt/Stark profiles with different assumptions. The repo currently has multiple
"truth" implementations rather than one validated kernel shared across contexts.

### F5. Retrieval discards amplitude information too early

Cosine similarity, area normalization, and L2 normalization are useful for some
tasks, but they are currently treated as default retrieval behavior. That makes
the manifold insensitive to absolute-intensity cues that should help infer `n_e`,
path length, throughput, and optical-depth effects.

### F6. Some correction modules are empirical placeholders, not model-backed
corrections

Matrix effects and parts of the non-ideal correction stack are represented by
hard-coded empirical factors or simplified escape-factor models. These are valid
placeholders, but they should be clearly separated from physics-backed models and
carried with provenance.

### F7. Validation does not yet enforce quantitative physical agreement

The test suite checks non-negativity, monotonicity, and basic scaling. It does
not yet establish cross-model agreement, recovery accuracy, reference parity, or
acceptable error envelopes for key observables such as line ratios, widths,
ionization fractions, and synthetic inversion recovery.

## Goals

1. Make plasma-state semantics explicit and consistent across the forward and
   inverse pipelines.
2. Establish one validated forward-physics kernel family shared by classic,
   Bayesian, and manifold workflows.
3. Preserve physically informative observables during retrieval and inversion.
4. Improve non-ideal plasma handling without hiding approximation boundaries.
5. Add quantitative validation targets and performance budgets.

## Non-Goals

1. Re-architecting the package layout.
2. Replacing all empirical models with first-principles collisional-radiative
   physics in one pass.
3. Building a fully non-LTE solver in this update.
4. Redesigning the CLI or documentation site structure.

## Quality Gates

These commands must pass for every implementation bead unless a bead is
documentation-only:

- `ruff check cflibs/ tests/`
- `black --check cflibs/ tests/`
- `JAX_PLATFORMS=cpu pytest tests/test_plasma.py tests/test_radiation.py tests/test_physics_invariants.py -v`

For inversion-focused beads, also include:

- `JAX_PLATFORMS=cpu pytest tests/test_boltzmann.py tests/test_solver.py tests/test_bayesian.py tests/test_hybrid_inversion.py -v`

For manifold-focused beads, also include:

- `JAX_PLATFORMS=cpu pytest tests/test_manifold.py tests/test_manifold_physics.py tests/test_vector_index.py -v`

For performance-sensitive beads, also include:

- `pytest tests/test_benchmark.py -v`

## Workstreams

## WS1. Plasma State Semantics And Thermodynamic Closure

### WS1-01: Define canonical plasma state semantics
As a developer, I need a single documented definition for temperature,
electron-density, heavy-particle density, abundance, path length, and
composition so all forward and inverse models use the same physical state.

Acceptance Criteria:
- Introduce a shared specification for abundance and density semantics used by
  classic, Bayesian, and manifold paths.
- Provide conversion helpers between mass fractions, number fractions,
  heavy-particle densities, and element number densities.
- Remove ambiguous places where concentrations are multiplied directly by `n_e`
  without an explicit physical scale.

### WS1-02: Replace sequential ionization updates with a coupled multi-stage LTE solve
As a model developer, I need ionization balance to be solved as a coupled system
with an explicit charge-neutrality target so stage populations remain physically
consistent.

Acceptance Criteria:
- Solve neutral, singly ionized, and doubly ionized populations from one coupled
  system rather than sequential remainder splitting.
- Support a configurable number of ionization stages based on available data.
- Expose diagnostics for charge-neutrality residual and stage-population sum.

### WS1-03: Harden partition-function evaluation
As a spectroscopic modeler, I need partition-function evaluation to respect data
validity ranges and avoid undocumented constant fallbacks in production paths.

Acceptance Criteria:
- Validate polynomial coefficients against their supported temperature range.
- Use cached direct-sum fallback when polynomial fits are unavailable.
- Emit structured warnings or errors when a species falls back to approximate
  partition behavior.

### WS1-04: Fix spectroscopic state identity
As a line-population solver, I need upper-state populations to be indexed by a
stable level identity rather than energy alone so degenerate or rounded states do
not alias.

Acceptance Criteria:
- Add or derive a stable state identifier for energy levels and transitions.
- Replace `(element, stage, E_k)` population keys in forward paths.
- Add regression tests that distinguish multiple states with identical or
  near-identical energies.

## WS2. Forward Radiation Kernel Fidelity

### WS2-01: Unify forward-model kernels across classic, Bayesian, and manifold paths
As a maintainer, I need the repo to share one validated set of kernels for
population, emissivity, line-shape, and instrument response so the main forward
paths do not drift apart physically.

Acceptance Criteria:
- Identify and extract shared kernels for LTE populations, emissivity, and
  broadening.
- Remove duplicated physics logic where reasonable.
- Add equivalence tests across `SpectrumModel`, `BayesianForwardModel`, and
  `ManifoldGenerator` on matched assumptions.

### WS2-02: Replace legacy scalar broadening defaults with a physically parameterized stack
As a user, I need the default forward model to represent Doppler, Stark, and
instrument broadening consistently instead of relying on a global ad hoc sigma.

Acceptance Criteria:
- Make broadening configuration explicit and physically parameterized.
- Ensure manifold instrument FWHM configuration is actually used during
  generation.
- Preserve backwards-compatible legacy mode only as an opt-in compatibility path.

### WS2-03: Replace hard transition prefilters with error-bounded line pruning
As a spectrum synthesizer, I need weak-line pruning to be based on quantified
error budgets rather than a fixed relative-intensity cutoff that can bias
continuum-like weak-line sums and diagnostic ratios.

Acceptance Criteria:
- Replace fixed `min_relative_intensity` production defaults with pruning logic
  tied to expected emissivity contribution or local visibility.
- Add tests showing that weak-line pruning does not materially change target
  ratios or integrated intensity beyond a declared tolerance.
- Keep a high-fidelity mode with no pruning except explicit wavelength range
  filtering.

### WS2-04: Add optically thick hooks to the forward spectrum path
As a physicist, I need the forward model to support optional self-absorption or
escape-factor corrections so intensity-scale discrepancies are not forced into
composition or density parameters.

Acceptance Criteria:
- Add an optional non-ideal radiative transfer hook to the classic forward model.
- Support at least one documented escape-factor or self-absorption mode with
  clear limits.
- Keep the default optically thin path explicit and well documented.

## WS3. Classic CF-LIBS Inversion Correctness

### WS3-01: Re-derive and implement the ionic-to-neutral correction consistently
As an inversion developer, I need the Saha-Boltzmann line correction to be
re-derived from the state semantics introduced in WS1 so ionic and neutral lines
share a physically consistent Boltzmann plane.

Acceptance Criteria:
- Document the exact transformed regression variable for neutral and ionic lines.
- Recompute the correction from the canonical plasma state.
- Add tests on synthetic data with known temperature and mixed ionization stages.

### WS3-02: Rework closure and pressure-balance updates around explicit species densities
As an analyst, I need closure and `n_e` updates to operate on physically
interpretable densities and concentrations instead of mixing normalization,
abundance, and charge-balance terms.

Acceptance Criteria:
- Separate abundance normalization from thermodynamic state updates.
- Compute electron-density updates from explicit species populations and an
  auditable pressure/closure model.
- Add convergence diagnostics that report which constraint is limiting.

### WS3-03: Add robust convergence and failure diagnostics
As a user, I need the classic solver to explain whether failures come from line
selection, partition data, thermodynamic inconsistency, or numerical stagnation.

Acceptance Criteria:
- Record per-iteration temperature, density, closure, and charge-balance
  residuals.
- Detect non-physical slope or intercept states before silently clamping.
- Expose termination reasons in the result object.

### WS3-04: Add line-quality screening tied to self-absorption and matrix sensitivity
As a spectroscopic workflow, I need the inversion to reject lines that are
systematically unsafe for Boltzmann fitting rather than trusting all line picks
equally.

Acceptance Criteria:
- Integrate self-absorption and line-quality metadata into the classic inversion.
- Provide configurable masks for resonance lines, saturated lines, and matrix-
  sensitive lines.
- Demonstrate improved recovery on synthetic or curated benchmark cases.

## WS4. Bayesian And Manifold Identifiability

### WS4-01: Decouple abundance from electron density in Bayesian and manifold forward models
As a probabilistic modeler, I need composition, heavy-particle density, and
electron density to be parameterized independently so the posterior does not
encode the wrong physical coupling.

Acceptance Criteria:
- Introduce an explicit heavy-particle density or emissivity-scale parameter.
- Remove `N_species_total = concentration * n_e` semantics from Bayesian and
  manifold forward models.
- Update priors, config objects, and parameter export accordingly.

### WS4-02: Preserve amplitude information in retrieval and coarse search
As an inversion workflow, I need manifold lookup modes that retain absolute or
semi-absolute scale information instead of defaulting to cosine-style shape-only
matching.

Acceptance Criteria:
- Add weighted chi-squared or noise-aware distance metrics.
- Support nuisance scaling parameters where scale mismatch is expected.
- Benchmark retrieval accuracy against cosine, correlation, and Euclidean modes.

### WS4-03: Rework spectral embeddings to retain physically meaningful structure
As a fast-search pipeline, I need embeddings that are useful for search without
 erasing density- and throughput-sensitive information by construction.

Acceptance Criteria:
- Evaluate alternatives to unconditional area normalization plus L2 normalization.
- Add ablation/performance tests for retrieval quality across temperature,
  density, and abundance perturbations.
- Keep approximate nearest-neighbor acceleration but make the representation
  physically motivated.

### WS4-04: Align hybrid inversion with the corrected manifold semantics
As a user of the hybrid inverter, I need coarse initialization and fine
optimization to operate in the same parameterization so the local optimizer does
not "repair" a biased manifold guess.

Acceptance Criteria:
- Update hybrid inversion packing/unpacking and loss definitions to the new
  semantics.
- Add consistency tests between manifold seed, Bayesian forward model, and
  optimized result.
- Quantify recovery error before and after the semantic fix.

## WS5. Non-Ideal Plasma Corrections

### WS5-01: Upgrade self-absorption correction beyond the Gaussian single-zone assumption
As a user, I need self-absorption correction modes that better reflect Voigt
profiles and stratified plasmas so strong-line corrections are not overtrusted.

Acceptance Criteria:
- Keep the current escape-factor model as a documented baseline.
- Add at least one improved mode that is Voigt-aware or explicitly two-zone.
- Add tests showing when correction should switch to masking rather than
  extrapolation.

### WS5-02: Replace hard-coded matrix-effect defaults with provenance-backed data
As a quantitative analysis workflow, I need matrix-effect corrections to carry
their source, valid range, and uncertainty rather than relying on undocumented
default tables.

Acceptance Criteria:
- Store correction provenance and calibration range alongside each factor.
- Separate example/demo factors from production defaults.
- Add validation that renormalization does not hide physically impossible output.

### WS5-03: Add explicit LTE validity guardrails to inference paths
As a diagnostician, I need forward and inverse workflows to report when LTE is
only weakly supported so users do not over-interpret parameter estimates.

Acceptance Criteria:
- Surface McWhirter validity metrics and margins in classic, Bayesian, and
  temporal workflows.
- Distinguish "soft penalty" from "physics satisfied" in outputs.
- Add tests at good-LTE, marginal-LTE, and invalid-LTE operating points.

## WS6. Validation, Benchmarking, And Performance

### WS6-01: Build quantitative physics reference sets
As a maintainer, I need curated reference cases for ionization fractions, line
ratios, Stark widths, and representative spectra so changes can be checked
against known targets.

Acceptance Criteria:
- Add reference datasets with provenance.
- Define allowed tolerances for each target observable.
- Cover at least one parity subset for spectrum shape and one for plasma-state
  diagnostics.

### WS6-02: Add cross-model equivalence tests
As a library, I need tests that ensure the classic forward model, Bayesian
forward model, and manifold generator agree when run under matched assumptions.

Acceptance Criteria:
- Compare spectra, line intensities, and width trends under aligned settings.
- Fail when semantic drift appears across implementations.
- Include both CPU and JAX-backed coverage where practical.

### WS6-03: Add recovery and identifiability benchmarks
As an inference library, I need end-to-end synthetic recovery benchmarks that
measure not just fit quality but whether the correct plasma state is
recoverable.

Acceptance Criteria:
- Benchmark temperature, density, and abundance recovery error.
- Include ambiguity cases where shape-only metrics should fail.
- Report which parameters are identifiable under each retrieval mode.

### WS6-04: Profile and optimize the dominant numerical kernels
As a performance-oriented codebase, I need to know which kernels dominate runtime
and memory so optimizations target the real bottlenecks instead of cosmetic
vectorization.

Acceptance Criteria:
- Profile classic, Bayesian, and manifold kernels on representative workloads.
- Optimize the top hotspots with documented before/after measurements.
- Keep physics equivalence tests green after each optimization.
