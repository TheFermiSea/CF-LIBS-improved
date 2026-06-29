# CF-LIBS Overhaul — Target Architecture

**Author:** Lead Architect (synthesis pass)
**Date:** 2026-06-25
**Inputs:** verified/*.md (adversarially-verified findings), literature/*.md checklists,
reference/*.md (ExoJAX/RADIS/RASCAL/specutils patterns)
**Status:** TARGET STATE — this document supersedes all inline per-milestone architecture notes in
BLUEPRINT.md and provides the stable anchor for Gaps 7–10 (see BLUEPRINT-ADDENDUM.md).

---

## 0. Guiding Principles

1. **Physics-only constraint is inviolable.** No ML library (`sklearn`, `torch`, `tensorflow`,
   `keras`, `flax`, `equinox`, `transformers`, `jax.nn`, `jax.experimental.stax`) in shipped
   `cflibs/`. Evolution tooling in `cflibs/evolution/` is the sole exception for LLM-driven
   optimization — but `cflibs/evolution/` itself may not import ML either (ruff TID251 + AST scan).

2. **Mission priority governs ordering:** ACCURACY > PRECISION > RELIABILITY > latency. Every
   performance-only item is deferred behind correctness items in the same area.

3. **Single source of truth for every physics quantity.** One canonical function/constant for
   emissivity, IPD, Voigt FWHM, Saha constant, molar-mass conversion, air↔vacuum wavelength. All
   other call sites import from that canonical location; no local copies.

4. **JAX optional, graceful degradation.** JAX is an accelerator, not a requirement. Every physics
   computation must have a working NumPy path. JAX paths must be parity-tested against NumPy
   (radiation F3, manifold F5, inv-solve F6/F10).

5. **Configuration over environment variables.** Typed, logged, gated config replaces the 15+
   `CFLIBS_*` env-var sprawl. Effective values surface in result metadata so benchmarks are
   reproducible. (See §5.)

---

## 1. Target Module Map

### 1.1 Canonical package boundaries

```
cflibs/
├── domain/                ← NEW: pure-physics data structures with no import side-effects
│   ├── result.py          ← CFLIBSResult (moved from inversion/solve/iterative.py:94)
│   ├── observation.py     ← LineObservation, BoltzmannFitResult (canonical; shim in
│   │                          inversion/common/data_structures.py kept for back-compat)
│   └── identification.py  ← ElementIdentification, IdentifiedLine + intensity_uncertainty slot
│
├── atomic/                ← atomic data: DB, structures, snapshot, partition functions
│   ├── database.py        ← AtomicDatabase + AtomicDataSource ABC (already here)
│   ├── structures.py      ← Transition, EnergyLevel, StarkData (already here)
│   ├── snapshot.py        ← AtomicSnapshot pytree (MOVED from core/jax_runtime.py:429–551)
│   └── partition.py       ← polynomial PF + direct summation + truncation invariant
│
├── plasma/                ← plasma state, Saha-Boltzmann solver, LTE validity
│   ├── state.py           ← SingleZoneLTEPlasma, TwoRegionPlasma
│   ├── saha_boltzmann.py  ← SahaBoltzmannSolver (canonical Saha + IPD)
│   ├── lte_validator.py   ← McWhirter + Cristoforetti; resonance-ΔE path default
│   └── constants.py       ← plasma physical constants (fold into core/constants.py)
│
├── radiation/             ← forward model: emissivity, broadening, spectrum assembly
│   ├── emissivity.py      ← THE canonical emissivity function (§2)
│   ├── profiles.py        ← Voigt/Gaussian/Lorentz; single OLIVERO_A/B constant pair
│   ├── stark.py           ← Stark broadening (radiation path)
│   ├── kernels.py         ← JAX-jittable kernels; delegates to emissivity.py
│   └── spectrum_model.py  ← SpectrumModel orchestrator; default BroadeningMode = PHYSICAL_DOPPLER
│
├── instrument/            ← instrument models; single response-curve implementation
│   ├── model.py           ← InstrumentModel + InstrumentModelJax (fixed from_file, F2)
│   ├── convolution.py     ← apply_instrument_function; sigma=0 guard (F10); exact 2√(2 ln2)
│   └── kernels.py         ← JIT-compiled instrument broadening (already here)
│
├── core/                  ← constants, config, JAX runtime, logging, cache
│   ├── constants.py       ← SAHA_CONST_CM3, KB_EV, CM_TO_EV, MCWHIRTER_CONST,
│   │                          OLIVERO_A=0.5343, OLIVERO_B=0.2169, FWHM_TO_SIGMA=1/2√(2ln2)
│   ├── config.py          ← typed Config (replaces env-var sprawl; §5)
│   ├── jax_runtime.py     ← JAX backend detection ONLY (AtomicSnapshot moved out)
│   ├── abc.py             ← SolverStrategy, InstrumentModelProtocol (TYPE_CHECKING import only)
│   └── cache.py           ← LRUCache with MISS sentinel (not None); key by id()+db_sha
│
├── inversion/
│   ├── common/            ← shared data structures, re-exports for back-compat
│   │   ├── data_structures.py  ← re-exports from domain/ + local PCA, IdentifiedLine
│   │   └── element_id.py
│   ├── preprocess/        ← signal preprocessing: SPLIT into sub-modules (§6)
│   │   ├── baseline.py
│   │   ├── noise.py       ← NoiseModel: shot noise on clean signal; variance from params (F3-rev)
│   │   ├── peaks.py       ← detect_peaks with sub-pixel centroiding (inv-preprocess F2)
│   │   ├── calibration/   ← wavelength calibration split from 2119-line monolith (F5)
│   │   │   ├── ransac.py  ← normalized RANSAC (inv-preprocess F1/F2)
│   │   │   ├── segmented.py
│   │   │   └── hough.py
│   │   └── outliers.py    ← moved to common/ (inv-preprocess F6) or kept here with re-export
│   ├── physics/           ← Saha-Boltzmann plane, closure, self-absorption, line selection
│   ├── identify/          ← ALIAS, comb, correlation, NNLS identifiers
│   ├── solve/             ← iterative, closed_form, bayesian, joint, manifold solvers
│   └── runtime/           ← DAQ streaming, temporal gate optimization
│
├── manifold/              ← JAX-accelerated manifold generation; single canonical forward
├── benchmark/             ← composition metrics, harness, ID benchmark
├── validation/            ← round-trip, NIST parity; solver-injection seam (F5)
├── hpc/                   ← distributed MCMC, SLURM integration
├── pds/                   ← PDS corpus; dead-end schema decision (§8)
├── io/                    ← exporters with unit labels + basis annotation
├── cli/                   ← CLI; unified --elements parsing
└── evolution/             ← blocklist scanner + driver config (no ML imports anywhere)
```

### 1.2 Domain-type seam: where CFLIBSResult, LineObservation, AtomicSnapshot live

**Motivation:** inv-solve F5 (CFLIBSResult in iterative.py imported by io/jitpipe/closed_form),
inv-common F5/F11 (LineObservation via physics/boltzmann shim, 11+ importers), core F1
(AtomicSnapshot domain carrier in jax_runtime.py).

**Decision:** Create `cflibs/domain/` as a leaf package (no cflibs imports except
`cflibs.core.constants`). Move:

- `CFLIBSResult` → `cflibs/domain/result.py` (M1-1)
- `LineObservation`, `BoltzmannFitResult` → `cflibs/domain/observation.py` (M1-2);
  `inversion/common/data_structures.py` re-exports for back-compat
- `AtomicSnapshot` → `cflibs/atomic/snapshot.py` (M1-4); pytree registration moves with it;
  `core/jax_runtime.py` retains only backend/capability detection code

**Rationale:** following RADIS's loose-coupling principle (database objects separate from
computation objects; reference RADIS §1.1 factory hierarchy). A domain leaf package ensures
the result contract can be imported by io/, jitpipe/, benchmark/, validation/ without pulling
in solver internals.

---

## 2. The Single Canonical Forward Model

### 2.1 Design decision — radiation/manifold forward consolidation (M6-4)

**Context:** Three parallel forward-physics implementations exist (manifold F5; radiation F3):
- `generator.py` — static `@jit` kernels with T^(-α) Stark scaling
- `batch_forward.py` — `single_spectrum_forward` without T^(-α)
- `basis_library.py` — CPU sequential, area-normalized, NNLS context

These diverge in confirmed ways (manifold F3: 50% Stark width error in batch_forward; manifold
F4: missing hc/4π in basis_library, benign only because area normalization cancels it).

**Decision:** Do NOT fully unify into one JAX kernel (the three serve different API shapes:
batch generation vs. single-spectrum vs. basis decomposition). Instead:

1. **Define `cflibs/radiation/emissivity.py`** as the single source of truth for the
   population-weighted emissivity formula:
   ```
   ε_ki = (hc / 4π λ_ki) · A_ki · N_s · (g_k / U(T)) · exp(-E_k / T_eV)
   ```
   (saha-boltzmann-lte literature §1.2; broadening-rt literature §1.1).
   Both the NumPy and JAX paths call into this module for the physics formula; only the
   vectorization strategy differs.

2. **`batch_forward.py`** must import and use the same T^(-α) Stark formula as `generator.py`
   (M6-3). Add parity test (M0-3) and flip xfail→assert when fix lands (GAP-15).

3. **`basis_library.py`** is intentionally area-normalized and NNLS-context-specific; document
   this explicitly (manifold F4). It does NOT need hc/4π since normalization cancels it.

4. **Parity-test binding** (M0-3) is mandatory: any change to one path triggers the parity suite
   for all three paths.

**ExoJAX pattern adopted (reference/exojax.md §2.2):** separate the one-time reference
computation (S0 at Tref) from the per-T scaling. In cflibs terms: pre-compute population weights
at `AtomicSnapshot` build time; scale by Boltzmann/Saha at each T during generation/solve.

### 2.2 Instrument response-curve consolidation (instrument F5)

**Context:** Three response-curve interpolation implementations with divergent edge handling:
`model.py:177` (scipy interp1d, fill=0), `kernels.py:52` (jnp.interp + jnp.where), and
`inversion/preprocess/response_correction.py` (independent).

**Decision (DESIGN-DECISION — see ADDENDUM DA-1):** Designate `instrument/kernels.py` as the
canonical interpolation path (JAX-compatible, jit-cached). `model.py` delegates to it via a
numpy-compatibility shim. `response_correction.py` is audited for divergence and either
redirected or documented as a separate use case. A single fill_value=0.0 policy is enforced at
all edges. Single parity test added before any change lands.

### 2.3 BroadeningMode default (radiation F2 + GAP-8)

**Context:** `BroadeningMode.LEGACY` (formula `0.01 * sqrt(T/0.86)`) is the default with no
DeprecationWarning. It is physically unmotivated for LIBS.

**Decision (DESIGN-DECISION — see ADDENDUM DA-2):** Promote `PHYSICAL_DOPPLER` as the new
default in a BENCHMARK-GATED pass. Add `DeprecationWarning` when LEGACY is selected (SAFE-NOW).
The exact default mode is decided by benchmark comparison on the held-out node set (M0-2).

**Instrument broadening per-wavelength:** for resolving-power instruments, use per-wavelength
sigma at each spectral bin (σ = λ/R/2√(2 ln2)), not the midpoint scalar. This is SAFE-NOW for
non-production non-default instrument paths (radiation F4, instrument F5).

---

## 3. Configuration Model

### 3.1 Replacing the env-var sprawl

**Context:** 15+ `CFLIBS_*` env vars (M7-7 in blueprint), 8 inline `CFLIBS_FF_*` flags in
benchmark (benchmark F6), `enforcement_mode` config fiction in evolution (evolution F2.1),
`evaluation_timeout_s=5.0` default in evolution (evolution F4.1). Truthy parsing is inconsistent
(`CFLIBS_USE_LAX_WHILE_LOOP` only accepts "1", `CFLIBS_RELIABILITY_FROM_UNCERTAINTY` accepts
"1/true/yes/on" — inv-solve F9/FA1).

**Target:** a single `cflibs.core.config.CFLIBSConfig` dataclass (frozen, validated on
construction) that replaces all env-var reads. Effective configuration is:
- Loaded once at solver/pipeline construction
- Logged at INFO level with all effective values
- Embedded in `CFLIBSResult.metadata` so benchmark runs are reproducible
- Feature-flag promotions are mechanical config changes, not env-var toggles

**Migration path:**
1. Create `CFLIBSConfig` with typed fields mirroring all current env vars (SAFE-NOW).
2. Constructor defaults match current env-var defaults (no behavior change on construct).
3. `_feature_flag()` helper reads from `CFLIBSConfig` not `os.environ` (one source of truth).
4. CLI and `cflibs generate-*` pass `CFLIBSConfig` through to all sub-components.
5. Env vars deprecated but still honored via adapter for backward compat (one release cycle).

**Note:** `CFLIBS_REFUSE_TO_REPORT`, `CFLIBS_MCWHIRTER_RESONANCE_DE`, and
`CFLIBS_USE_LAX_WHILE_LOOP` are the three highest-value flags to migrate first; they gate the
M2/M3/M4 accuracy items.

---

## 4. Noise and Uncertainty Model

### 4.1 Physical noise model (Poisson + readout; signal-dependent variance)

**Context:** `round_trip.py:588–615` adds background before shot noise, applies Poisson to
signal+background (inflating variance), then uses the noise-realized intensity as its own
variance estimate (validation F3-revised, REAL/HIGH). The Bayesian forward model uses a
Gaussian likelihood with fixed σ (Pearson bias, inv-solve F4). Both violate the Poisson+readout
model mandated by bayesian-oe literature §1.4.2.

**Target `NoiseModel` (canonical):**
```python
# Clean signal (before background addition):
var_shot = max(clean_signal, 0)          # Poisson: Var[photons] = E[photons]
var_total = var_shot + sigma_readout**2  # Gaussian readout added in quadrature
# Then add background to intensity (AFTER variance is computed):
noisy = clean_signal + background + rng.normal(scale=sqrt(var_total))
```

**Bayesian likelihood:** pixel-dependent variance `σ_k²(x) = λ_k(x) + σ_RON²` evaluated at each
MCMC proposal (bayesian-oe §2.4). Include `log(σ_k²)` normalization term for BIC/WAIC model
comparison (bayesian-oe §2.5, GAP-13).

**Signal-dependent variance** is the gate condition: if max(SNR_per_line) < 100 for any line
used in a Bayesian fit, fixed-σ approximation is inaccurate. The threshold is configurable via
`CFLIBSConfig`.

### 4.2 Uncertainty propagation chain

The complete propagation is: atomic-data uncertainty (aki_uncertainty on Transition) → preserved
through `to_line_observations` (inv-common F2) → carried in `LineObservation.aki_uncertainty` →
used by `_build_sigma_y` in boltzmann fitter → propagated to `BoltzmannFitResult.cov` → through
`CFLIBSResult.uncertainties`. The gap is at the `IdentifiedLine.intensity_uncertainty` slot
(inv-common F12, NEW finding): adding this field unblocks the propagation from line-detection
through the result hierarchy (inv-common F1/F2/F3, GAP-E).

---

## 5. Partition-Function ↔ Saha IPD Truncation-Consistency Invariant

**This is identified as CRITICAL in saha-boltzmann-lte literature §P2 and GAP-12.**

**The invariant:** The partition-function sum cutoff (E_max = χ - Δχ) MUST use the same Δχ
as the Saha equation exponent (χ_eff = χ - Δχ). Breaking this makes the Saha equation
internally inconsistent: the forward model produces n_II/n_I ratios that do not correspond
to the population calculation in the Boltzmann plot.

**Current risk:** radiation F3 documents two divergent population code paths in `kernels.py`
(precomputed vs. kernel-internal). If these two paths use different IPD cutoffs, they will
produce different Saha ratios for the same (T, n_e, composition). The audit in verified/radiation
F3 notes the `_directsum_partition_functions` was added to close the gap — but there is no
invariant test.

**Architectural enforcement:**
1. A single `compute_ipd(n_e, T, model)` function in `cflibs/plasma/` is the only site that
   computes Δχ. All callers (Saha solver, partition function truncation, level population filter)
   must import and call this function.
2. The Saha solver (`saha_boltzmann.py`) passes the computed Δχ to the partition function
   evaluator as an explicit argument; the evaluator does NOT independently recompute IPD.
3. An invariant test (M0-1 extended; GAP-12) asserts:
   - PF sum cutoff == χ - Δχ for both polynomial and direct-sum paths
   - Saha ratio computed via `cflibs.plasma.saha_boltzmann` agrees within 0.1% with
     independently-derived value at (n_e=1e17, T=1e4 K) for Fe I/II

---

## 6. Air ↔ Vacuum Wavelength Convention (Cross-Cutting Invariant)

**Context:** NIST ASD reports air wavelengths above 200 nm. Atomic constants (Doppler broadening,
Stark formula) and the physics-plane equations (Boltzmann y-value = ln(I·λ/(gA))) require
consistent λ. A 0.05–0.1 nm systematic air/vacuum offset would dominate the sub-pixel centroid
gains of M7-8 and create false identification peaks at tolerances below ~0.1 nm.

**Status:** No verified finding examined this (GAP-11 in CRITIC). Audit required before any
wavelength-calibration or sub-pixel centroiding change goes in.

**Convention target (to be confirmed by audit):**
- DB stores wavelengths in **air** (matching NIST/ASD provenance; see datagen_v2.py and
  ingest_kurucz_atomic.py for confirmation of provenance)
- Line matching tolerance is in **air nm**
- Forward model emissivity uses the stored (air) λ for the hc/λ factor
- Conversion to vacuum is applied ONLY if an explicitly-vacuum detector calibration is used
- A single `air_to_vacuum(wl_nm)` / `vacuum_to_air(wl_nm)` utility lives in `cflibs/core/`
  using the Edlén (1953) formula (same as NIST)
- Every module that stores or compares wavelengths annotates the convention in its docstring

**Audit command (from CRITIC GAP-11):**
```bash
rg -ni "vacuum|air.?wavelength|n_air|edlen|ciddor" cflibs/
```

---

## 7. Preprocessing Module Split (GAP-7)

**Context:** `wavelength_calibration.py` is a 2119-line monolith (inv-preprocess F5, confirmed
HIGH) containing six distinct subsystems: quality-gate, Hough transform (273–369), RANSAC
(431–643), line-pool construction (646–815), candidate pairs (818–853), segmented orchestration
(1503–1748). M7-2 and M7-8 touch RANSAC math inside the monolith; splitting is a prerequisite
for safely isolating and flag-gating those paths.

**Target split (SAFE-NOW, behavior-preserving):**
```
inversion/preprocess/calibration/
    __init__.py        ← re-exports public API
    ransac.py          ← _ransac_search, _ransac_fit, _refine_robust_inliers
    hough.py           ← detect_ccd_seams, Hough logic
    pool.py            ← line pool construction, candidate pairs
    quality.py         ← _quality_gate_check, coverage gates
    segmented.py       ← calibrate_wavelength_axis_segmented (orchestrator)
```

**Acceptance:** import-parity test (all public functions accessible from original path);
existing calibration test suite green on the split module. No behavior change.

**`outliers.py` relocation (inv-preprocess F6):** Move to `inversion/common/outliers.py` and
re-export from `inversion/preprocess/outliers.py`. The module is broadly re-exported at
`cflibs.inversion` level; the move must update `__init__.py` re-exports.

---

## 8. PDS Schema: Dead-End Bridge Decision (GAP-9)

**Context:** `PDSValidationDataset` in `pds/validation.py` is confirmed unwired from benchmark
and validation pipelines despite its docstring claiming it is. The docstring claim is currently
false (pds F2-2, REAL/MEDIUM).

**Decision:** Wire or delete within M3/M5 window.
- If `PDSValidationDataset` is wired into the benchmark adapter during M3 work (it participates
  in the mole/mass fraction correction), keep it and update the docstring.
- If it is not wired by the time M3 lands, mark it `# DEPRECATED: will be removed in next major
  version` and file a removal issue.

**Data-correctness fix (GAP-9 pds F1-2):** SCCT5/SCCT7 compositions are copy-pasted from
CCCT3/CCCT2 with K silently dropped. Fix: verify against ChemCam published compositions
(Fabre et al. 2011 and companion papers) and restore K with a citation comment. This is
SAFE-NOW data-truth correction, not a code logic change.

---

## 9. Rust ↔ Python Comb Interface (GAP-10 + native-rust F2)

**Context:**
- `kdet_filter_elements` Rust branch is dead under default `shift_coherence_veto=True` pipeline
  (native-rust F2, REAL/HIGH). The comment in `jitpipe/host.py:1148` confirms this is an
  intentional architectural gap.
- Greedy matching orientation is transposed between Rust (`outer=peaks`) and Python
  (`outer=transitions`) (native-rust Missed-A, REAL/MEDIUM). Diverges on ambiguous inputs.
- F1 tie-break epsilon differs: Rust 1e-12 vs Python np.isclose rtol=1e-5 (native-rust Missed-B).

**Decision (DESIGN-DECISION — see ADDENDUM DA-3):**
- The dead kdet branch is intentional. Options: (a) port the density-score path to Python and
  remove the dead Rust branch, (b) document as "future Rust acceleration" with a `TODO(kdet)`.
  Decision requires a measured comparison of ID F1 with and without the kdet filter enabled
  (impossible to measure currently since the branch is dead — fixing the gate condition to
  enable it is the first step).
- Greedy orientation MUST be aligned before the parity fixture (M0-3) can be flipped from
  xfail to assert. The Python semantics (outer=transitions) are the original design and should
  be treated as ground truth.
- F1 tie-break: adopt the Python `np.isclose` semantics in Rust (change `1e-12` threshold to
  match Python's default relative tolerance).

---

## 10. Plasma Architecture: `plasma/` Module Roles (plasma F6 + GAP-E)

`plasma/` is the home for the closed-form forward Saha-Boltzmann physics that is shared by both
the forward model and the inversion. It should NOT import from `inversion/` (currently clean).
The architecture target keeps:

- `plasma/state.py` — plasma state dataclass; `SingleZoneLTEPlasma` + `TwoRegionPlasma`
- `plasma/saha_boltzmann.py` — Saha equation + multi-stage charge balance + IPD
- `plasma/lte_validator.py` — McWhirter + Cristoforetti; resonance-ΔE is default (post-M2)
- `plasma/partition.py` — polynomial PF; direct-sum PF; truncation-consistency invariant

**`TwoRegionPlasma` (plasma F6):** documented as UNVALIDATED / EXPERIMENTAL with explicit
`UserWarning` on construction when `two_region=True`. The magic-number 0.8/0.3-0.7 constants
and element-scope list (`{"Si","Fe","Ca","Al","Mg"}`) must be documented with their absence of
literature basis (DESIGN-DECISION M5-8 in blueprint).

---

## 11. Key Verified Design Decisions Summary

| id | decision | class | where decided |
|----|----------|-------|---------------|
| DA-1 | Response-curve single canonical path: instrument/kernels.py | DESIGN-DECISION | §2.2 |
| DA-2 | BroadeningMode default: PHYSICAL_DOPPLER (benchmark-gated flip) | DESIGN-DECISION | §2.3 |
| DA-3 | Rust kdet branch: port vs. document-as-future | DESIGN-DECISION | §9 |
| DA-4 | Air/vacuum convention: DB stores air; single conversion utility | AUDIT-FIRST | §6 |
| DA-5 | CFLIBSConfig typed config replaces env-var sprawl | SAFE-NOW | §3 |
| DA-6 | IPD/PF truncation consistency enforced by single compute_ipd() call | SAFE-NOW | §5 |
| DA-7 | NoiseModel: shot noise on clean signal; variance from params | SAFE-NOW | §4 |
| DA-8 | domain/ leaf package for CFLIBSResult/LineObservation/AtomicSnapshot | SAFE-NOW | §1.2 |

---

## 12. What This Architecture Does NOT Change

- The CF-LIBS algorithmic flow (SahaBoltzmann + Boltzmann-plot + closure equation + n_e iteration)
  is preserved and validated. Any physics change is benchmark-gated.
- JAX is still optional. The architecture adds no new JAX dependencies.
- `cflibs/evolution/` remains a separate optimization tooling package. The physics-only constraint
  is not relaxed.
- The CLI interface and YAML config schema are backward-compatible through the CFLIBSConfig
  migration (env vars honored for one release cycle).
