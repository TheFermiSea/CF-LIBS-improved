# Manifold-Accelerated Forward-Model Element Identification

## Design Document — CF-LIBS Next-Generation Analysis Pipeline

**Author:** Brian Squires
**Date:** 2026-03-16
**Epic:** CF-LIBS-MANIFOLD
**Status:** Draft

---

## 1. Executive Summary

This document describes a next-generation element identification pipeline for
Calibration-Free LIBS that replaces peak-matching (ALIAS) with full-spectrum
forward-model fitting accelerated by GPU vector similarity search. The approach
combines the physics correctness of iterative CF-LIBS (Hermann et al.) with the
speed of pre-computed spectral libraries indexed via FAISS.

**Target application:** Femtosecond/picosecond LIBS for metal and ceramic
additive manufacturing (AM) process monitoring.

**Current performance (ALIAS):** 77% recall, 24% precision on 49 Aalto spectra.

**Target performance:** >95% recall, >85% precision, <200ms per spectrum.

---

## 2. Motivation and Problem Statement

### 2.1 Why Peak-Matching Fails

The ALIAS algorithm (Noël et al. 2025) identifies elements by matching detected
spectral peaks to NIST database wavelengths. This approach suffers from
fundamental limitations:

1. **False positives from coincidental wavelength matches.** Elements with dense
   line spectra (rare earths, transition metals) produce coincidental matches
   against peaks that actually belong to other elements. Our benchmark shows 289
   false positives across 49 spectra — the majority from Na, O, K, N (atmospheric
   contaminants) and Mg, Ca, Fe (ubiquitous trace elements).

2. **Peak matching discards most spectral information.** A typical LIBS spectrum
   has 4096+ wavelength channels, but peak matching only uses ~50-100 peak
   positions and intensities. The shape, width, and wing structure of lines —
   which encode T, nₑ, and self-absorption — are ignored.

3. **No physical consistency enforcement.** ALIAS scores each element
   independently. There is no constraint that the detected element set must
   collectively explain the observed spectrum at a single, self-consistent (T, nₑ).

### 2.2 The Forward-Model Alternative

Hermann et al. (2019, Analytical Chemistry) demonstrated that CF-LIBS achieves
far better accuracy when the entire observed spectrum is modeled as the spectral
radiance of an LTE plasma with known composition. Their iterative algorithm
optimizes n+2 parameters (T, nₑ, n-1 element fractions, plasma diameter L)
to minimize the residual between computed and measured spectra [Hermann 2019].

The OPSIAL software (Tan 2018) implements this as a line-by-line radiative
transfer solver that was successfully applied to ChemCam/Mars data, including
a two-temperature model (T_exc=13500K, T_i=9000K) for partial-LTE conditions
[Tan 2018, OPSIAL].

However, iterative full-spectrum fitting from scratch is slow (~seconds per
spectrum) and sensitive to initial conditions. Our approach accelerates this
with pre-computed spectral libraries.

### 2.3 The fs/ps-LIBS Simplification

Femtosecond and picosecond laser pulses produce cooler, less dense plasmas than
nanosecond LIBS. Key implications:

- **Temperature range:** 4,000–12,000 K (vs 8,000–20,000 K for ns-LIBS)
- **Ionization stages:** Only I and II (neutral and singly-ionized)
- **Optical thickness:** Generally optically thin — self-absorption is minimal
  for most lines. The ps-LIBS literature shows high linearity in Saha-Boltzmann
  plots, indicating the plasma approaches ideal optically thin conditions
  [CF-LIBS Review, Zhang et al. 2022].
- **LTE validity:** Shorter plasma lifetime means LTE is marginal, but the
  Boltzmann distribution typically holds for the lower excited states.

This simplifies the forward model: fewer ionization stages, smaller partition
function sums, minimal self-absorption correction needed.

### 2.4 Additive Manufacturing Context

In metal/ceramic AM, the alloy composition is arbitrary — not constrained by
geochemistry. Common systems include:

- **Titanium alloys:** Ti6Al4V, Ti-Nb-Zr, Ti-Mo
- **Nickel superalloys:** Inconel 718 (Ni-Cr-Fe-Nb-Mo-Ti-Al), Hastelloy
- **Steels:** 316L (Fe-Cr-Ni-Mo-Mn-Si), maraging steel (Fe-Ni-Co-Mo-Ti)
- **Aluminum alloys:** AlSi10Mg, Al-Cu, Al-Zn-Mg
- **Ceramics:** Al₂O₃, ZrO₂-Y₂O₃, WC-Co, SiC, TiN
- **High-entropy alloys:** CoCrFeMnNi, TiVNbMoTa

No geochemical prior is applicable. Any combination of metallic/ceramic
elements may appear. This rules out composition-enumeration approaches and
motivates the linear decomposition strategy described below.

---

## 3. Architecture

### 3.1 Overview: Retrieve → Decompose → Refine

```
                    OFFLINE (once, ~hours on GPU)
┌──────────────────────────────────────────────────────┐
│  Generate single-element basis library               │
│  68 elements × stages I,II × 50 T × 20 nₑ          │
│  = 68,000 synthetic spectra (4096 pixels each)       │
│  Store as HDF5 + FAISS GPU index                     │
└──────────────────────────────────────────────────────┘

                    ONLINE (per spectrum, ~100ms)
┌──────────────────────────────────────────────────────┐
│  Step 1: PREPROCESS                                  │
│  Baseline subtraction (PSO-arPLS or SNIP)            │
│  Area normalization                                  │
│  Noise estimation                                    │
│                                                      │
│  Step 2: RETRIEVE (T, nₑ) via FAISS                 │
│  PCA-embed observed spectrum → 30-dim vector         │
│  FAISS GPU search against 68k basis spectra          │
│  Median (T, nₑ) of top-50 neighbors                 │
│  → estimated plasma parameters in ~1ms               │
│                                                      │
│  Step 3: DECOMPOSE via full-spectrum NNLS            │
│  At estimated (T, nₑ), retrieve 68 basis spectra    │
│  B = [S_el1(λ), S_el2(λ), ..., S_el68(λ)]          │
│  Solve: min ||observed - B @ c||², c ≥ 0            │
│  Elements with c_i > threshold → detected            │
│  → element identification + initial concentrations   │
│                                                      │
│  Step 4: REFINE via JAX gradient optimization        │
│  Free parameters: T, nₑ, {c_i for detected els}     │
│  JAX-differentiable forward model                    │
│  L-BFGS-B from NNLS starting point                   │
│  → refined T, nₑ, concentrations                    │
│                                                      │
│  Step 5: VALIDATE via Bayesian model selection       │
│  BIC = n·ln(RSS/n) + k·ln(n)                        │
│  Compare fits with/without borderline elements       │
│  Boltzmann consistency check (R² > 0.95)             │
│  → final element list + uncertainties                │
└──────────────────────────────────────────────────────┘
```

### 3.2 Why Linear Decomposition Works for fs-LIBS

In an optically thin LTE plasma, the spectral radiance at wavelength λ is:

    L(λ) = Σᵢ cᵢ · Sᵢ(λ; T, nₑ)

where cᵢ is the number fraction of element i and Sᵢ(λ; T, nₑ) is the
single-element spectrum at (T, nₑ). This linearity holds when:

1. The plasma is optically thin (no self-absorption)
2. All elements share the same T and nₑ (LTE, uniform plasma)
3. Inter-element interactions are negligible (no matrix effects on excitation)

Conditions (1) and (2) are well-approximated in fs/ps-LIBS. Condition (3)
breaks down for matrix effects, but these are second-order corrections
addressable in the refinement step.

The NNLS decomposition exploits this linearity: given the observed spectrum
b and the basis matrix B (68 columns, 4096 rows), solve:

    min ||b - B·c||²  subject to  c ≥ 0

This is a convex optimization with a unique global minimum, solvable in ~1ms.
The non-negativity constraint ensures physically meaningful concentrations.

### 3.3 FAISS-Accelerated Plasma Parameter Estimation

Instead of the fragile single-element Boltzmann slope, we estimate (T, nₑ)
by finding the most similar synthetic spectra in the pre-computed library:

1. The 68,000 basis spectra are PCA-embedded to 30 dimensions and L2-normalized
2. A FAISS IVF-Flat GPU index enables sub-millisecond nearest-neighbor search
3. The top-50 neighbors vote on (T, nₑ) via weighted median
4. The weights are the inverse L2 distances

This is more robust than Boltzmann slope because:
- It uses the full spectral shape, not just a few lines
- It's immune to self-absorption artifacts on individual lines
- It averages over multiple elements' temperature signatures
- It naturally handles the case where no single element dominates

### 3.4 JAX-Differentiable Forward Model for Refinement

The existing `ManifoldGenerator._compute_spectrum_snapshot()` implements the
full forward model in JAX with @jit compilation. For the refinement step, we
need this to be differentiable with respect to (T, nₑ, concentrations).

JAX provides automatic differentiation via `jax.grad()` and `jax.jacobian()`.
The key computation graph:

```
(T, nₑ, c₁..cₙ)
    │
    ├─→ Saha equation: nᵢᵢ/nᵢ = f(T, nₑ, IP)
    │     Uses partition functions U(T) from polynomial coefficients
    │
    ├─→ Boltzmann populations: nₖ = nₛ · (gₖ/U) · exp(-Eₖ/kT)
    │
    ├─→ Line emissivities: εₖᵢ = (hc/4πλ) · Aₖᵢ · nₖ
    │
    ├─→ Line profiles: Voigt(λ; σ_Doppler, γ_Stark)
    │     σ_Doppler = λ·√(2kT/mc²)
    │     γ_Stark = w_ref · (nₑ/1e16) · (T/0.86)^(-0.5)
    │
    └─→ Spectrum: S(λ) = Σₖ εₖᵢ · V(λ-λₖ; σ, γ) ⊗ G_instrument
```

All operations are differentiable in JAX. The L-BFGS-B optimizer
(`jaxopt.LBFGSB`) can minimize the χ² residual:

    χ² = Σⱼ [(observed_j - model_j) / σ_j]²

where σ_j is the noise estimate at pixel j.

### 3.5 Bayesian Model Selection

The NNLS step may assign small but nonzero coefficients to spurious elements.
The refinement step may overfit by including too many free parameters. To
determine the optimal element set, we use the Bayesian Information Criterion:

    BIC = n·ln(RSS/n) + k·ln(n)

where n = number of spectral pixels, k = number of free parameters
(2 + N_elements for T, nₑ, and N_elements concentrations), and RSS is the
residual sum of squares.

**Procedure:**
1. Start with all NNLS-detected elements (c_i > threshold)
2. Compute BIC for this full model
3. For each element in decreasing order of NNLS coefficient:
   - Remove element, re-fit, compute BIC
   - If BIC decreases → element was spurious, keep it removed
   - If BIC increases → element is needed, restore it
4. Final element set = those that survive backward elimination

This naturally handles the precision problem: adding a spurious element
increases k without significantly decreasing RSS, so BIC penalizes it.

The Gemini Deep Research report additionally recommends a Boltzmann linearity
filter (R² > 0.95) as a physics-informed validation gate — if the detected
line intensities don't conform to ln(Iλ/gA) vs Eₖ linearity at the fitted
temperature, the identification is rejected [Gemini report, "Establishing
Algorithmic Benchmarks"].

---

## 4. Detailed Component Design

### 4.1 Single-Element Basis Library Generation

**Module:** `cflibs/manifold/basis_library.py` (new)

**Parameters:**
```python
@dataclass
class BasisLibraryConfig:
    db_path: str                        # Path to libs_production.db
    output_path: str                    # HDF5 output
    elements: List[str]                 # All DB elements
    ionization_stages: List[int] = (1, 2)   # fs-LIBS: I and II only
    wavelength_range: Tuple[float, float] = (200.0, 900.0)  # nm
    pixels: int = 4096
    temperature_range: Tuple[float, float] = (4000.0, 12000.0)  # K
    temperature_steps: int = 50
    density_range: Tuple[float, float] = (1e15, 5e17)  # cm⁻³
    density_steps: int = 20
    instrument_fwhm_nm: float = 0.05    # Spectrometer resolution
    use_voigt_profile: bool = True
    use_stark_broadening: bool = True
    batch_size: int = 1000              # GPU batch
```

**Output structure (HDF5):**
```
/basis_library.h5
├── spectra          (68, 1000, 4096)  float32  # [element, T×nₑ, wavelength]
├── params           (68, 1000, 2)     float32  # [element, T×nₑ, (T, nₑ)]
├── wavelength       (4096,)           float64  # nm
├── elements         (68,)             str      # element symbols
├── element_index    {str: int}        # element → index mapping
└── attrs:
    ├── temperature_range, density_range
    ├── ionization_stages
    ├── physics_version, instrument_fwhm_nm
    └── generation_timestamp
```

**Generation algorithm:**
```
for each element in elements:
    for each (T, nₑ) in grid:
        1. Solve Saha equation for stage I/II balance
        2. Compute Boltzmann populations for all levels
        3. Calculate line emissivities: ε = (hc/4πλ)·A_ki·n_k
        4. Render Voigt profiles at each line position
        5. Convolve with instrument response
        6. Store spectrum
```

This reuses the existing `ManifoldGenerator._compute_spectrum_snapshot()` but
in single-element mode. JAX vmap over the (T, nₑ) grid gives ~1000 spectra/s
on Apple Silicon, ~10,000/s on NVIDIA GPU.

**Total generation time:** 68 elements × 1000 grid points ÷ 1000 spectra/s
= ~68 seconds on Apple Silicon, ~7 seconds on NVIDIA.

### 4.2 FAISS Index Construction

**Module:** `cflibs/manifold/vector_index.py` (extend existing)

The existing `SpectralEmbedder` and `VectorIndex` classes handle this.
Extensions needed:

1. **Element-tagged index:** Each vector in FAISS carries metadata
   (element, T, nₑ) for neighbor voting.
2. **GPU index:** Use `faiss.index_cpu_to_gpu()` for sub-ms search.
3. **Embedding tuning:** The default PCA-30 may need adjustment.
   Test n_components ∈ {20, 30, 50, 100} on the Aalto benchmark.

**Index sizing:**
```
68,000 vectors × 30 dimensions × 4 bytes = 8.2 MB
FAISS IVF-Flat with nlist=256: ~8 MB + overhead
Fits entirely in GPU memory.
```

### 4.3 Full-Spectrum NNLS Identifier

**Module:** `cflibs/inversion/spectral_nnls_identifier.py` (new)

This is the core new component — replaces ALIAS for element identification.

```python
class SpectralNNLSIdentifier:
    """
    Element identification via NNLS decomposition of the full observed
    spectrum into single-element basis spectra.

    At a given (T, nₑ), the observed spectrum is modeled as:
        observed(λ) ≈ Σᵢ cᵢ · basis_i(λ; T, nₑ) + continuum(λ)

    where basis_i is the pre-computed synthetic spectrum of element i.
    NNLS enforces cᵢ ≥ 0. Elements with cᵢ above a significance
    threshold are reported as detected.
    """

    def __init__(
        self,
        basis_library: BasisLibrary,
        vector_index: VectorIndex,
        detection_snr: float = 3.0,     # c_i / σ(c_i) > 3 for detection
        bic_pruning: bool = True,
    ):
        ...

    def identify(
        self, wavelength: np.ndarray, intensity: np.ndarray
    ) -> ElementIdentificationResult:
        """Full pipeline: preprocess → estimate T,nₑ → NNLS → validate."""
        ...
```

**Key design decisions:**

- **Detection threshold:** Instead of absolute CL thresholds, use the
  statistical significance of each NNLS coefficient. The uncertainty σ(cᵢ)
  is estimated from the residual variance and the diagonal of (BᵀB)⁻¹.
  An element is detected if cᵢ / σ(cᵢ) > 3 (99.7% confidence).

- **Continuum handling:** Add a low-order polynomial (degree 3-5) as
  additional columns in the basis matrix to absorb any residual continuum
  not removed by baseline subtraction.

- **Interpolation between grid points:** The basis library has T and nₑ on
  a discrete grid. For estimated (T, nₑ) between grid points, bilinearly
  interpolate the basis spectra. This is fast (68 interpolations) and
  avoids the need to recompute the forward model.

### 4.4 Joint Refinement via JAX Optimization

**Module:** `cflibs/inversion/joint_optimizer.py` (extend existing)

The existing `JointOptimizer` uses softmax parametrization for concentrations.
Extend it to:

1. Accept the NNLS result as starting point (T₀, nₑ₀, c₀)
2. Only optimize concentrations for detected elements (reduce dimensionality)
3. Use JAX autodiff for gradient computation
4. Return refined (T, nₑ, concentrations) with covariance matrix

**Loss function:**
```
L(T, nₑ, c) = Σⱼ wⱼ · [observed_j - model_j(T, nₑ, c)]²
             + λ_T · (T - T_prior)²            # weak prior from NNLS
             + λ_n · (log nₑ - log nₑ_prior)²  # weak prior
```

where wⱼ = 1/σ²_j (inverse noise variance) and λ_T, λ_n are weak
regularization strengths to prevent divergence.

### 4.5 Bayesian Model Selection (BIC Pruning)

**Module:** `cflibs/inversion/model_selection.py` (new)

```python
def bic_prune_elements(
    wavelength: np.ndarray,
    observed: np.ndarray,
    basis_matrix: np.ndarray,     # (N_candidates, N_pixels)
    element_list: List[str],
    noise_estimate: np.ndarray,
) -> Tuple[List[str], np.ndarray]:
    """
    Backward elimination of elements using BIC.

    Returns (final_elements, final_concentrations).
    """
```

---

## 5. Data Flow and Storage

### 5.1 Offline Pipeline

```
libs_production.db (NIST atomic data)
    │
    ▼
BasisLibraryGenerator (JAX GPU)
    │
    ├─→ basis_library.h5     (68, 1000, 4096) = 1.1 GB
    │
    └─→ basis_index.faiss    (68k vectors × 30-dim) = 8 MB
         + metadata.h5       (element, T, nₑ per vector)
```

### 5.2 Online Pipeline (per spectrum)

```
Raw spectrum (wavelength, intensity)
    │
    ▼
Preprocessor
    ├─ Baseline: PSO-arPLS or SNIP (existing)
    ├─ Normalize: area normalization
    └─ Noise: MAD-based estimate (existing)
    │
    ▼
FAISS search → (T_est, nₑ_est)         [~1ms, GPU]
    │
    ▼
Basis interpolation → B(T_est, nₑ_est)  [~0.1ms]
    │
    ▼
NNLS decomposition → c_initial          [~1ms]
    │
    ▼
Significance filtering → detected_elements  [~0.1ms]
    │
    ▼
JAX refinement → (T, nₑ, c_refined)     [~50-100ms, GPU]
    │
    ▼
BIC pruning → final_elements             [~10ms]
    │
    ▼
ElementIdentificationResult
```

**Total latency: ~60-110ms per spectrum** (vs 28s for ALIAS)

---

## 6. Validation Strategy

### 6.1 Benchmark: Aalto Spectral Library

Run the same 49-spectrum benchmark used for ALIAS:
- 13 pure elements + 36 minerals with known compositions
- Measure recall, precision, F1 (standard and LIBS-aware)
- Compare directly to ALIAS baseline numbers

**Success criteria:**
- Recall > 90% (vs 77% ALIAS)
- Precision > 50% standard / >70% LIBS-aware (vs 24%/41% ALIAS)
- F1 > 65% standard (vs 36% ALIAS)

### 6.2 Synthetic Round-Trip Validation

Generate synthetic spectra with known compositions, add realistic noise,
and verify the pipeline recovers the correct elements and concentrations.
Use the existing `GoldenSpectrum` validation framework.

### 6.3 AM-Specific Validation

Test on spectra from common AM alloys:
- Ti6Al4V (Ti-6Al-4V)
- 316L stainless (Fe-Cr-Ni-Mo-Mn-Si)
- Inconel 718 (Ni-Cr-Fe-Nb-Mo-Ti-Al)

### 6.4 Speed Benchmark

Measure end-to-end latency per spectrum on:
- Apple Silicon (M-series, JAX Metal)
- NVIDIA GPU (JAX CUDA)
- CPU-only fallback

---

## 7. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Linearity assumption breaks (matrix effects) | Medium | High | Refinement step handles nonlinear corrections |
| Basis library doesn't cover all T/nₑ combinations | Low | Medium | Interpolation + refinement from nearest grid point |
| NNLS assigns nonzero coefficients to spurious elements | High | Medium | BIC pruning + SNR-based significance threshold |
| fs-LIBS plasma departs from LTE | Medium | Medium | Two-temperature model (T_exc ≠ T_ion) as in OPSIAL |
| Self-absorption on resonance lines of major elements | Low (fs-LIBS) | Low | Doublet-ratio SA correction as fallback |
| FAISS T/nₑ estimate inaccurate for unusual spectra | Low | Low | Falls back to Boltzmann slope; refinement corrects |

---

## 8. Planned Enhancements (from Gemini Deep Research Synthesis)

The following enhancements were identified by cross-referencing the Gemini Deep
Research report (71 websites surveyed, March 2026) with the academic literature
from Semantic Scholar. They are ordered by expected impact and feasibility.

### 8.1 Enhanced Signal Conditioning

**Multi-scale Wavelet Denoising (DWT with db5 filter)**

The current preprocessing uses SNIP baseline subtraction and MAD noise
estimation. For single-shot AM process monitoring where SNR is often low,
integrating Discrete Wavelet Transform (DWT) denoising with a 'db5' mother
wavelet would separate high-frequency stochastic noise from sharp atomic
transitions before the NNLS decomposition step. This is particularly important
for detecting trace alloying elements (e.g., V at 0.4% in Ti6Al4V) whose
lines may be buried in shot noise.

**Implementation:** `pywt.wavedec()` → threshold detail coefficients →
`pywt.waverec()`. Add as optional preprocessing step before NNLS.

**Adaptive Atmospheric Masking**

Since atmospheric contaminants (Na, O, K, N, H, Ar) are the dominant source
of false positives (accounting for ~55% of all FPs in the Aalto benchmark),
the preprocessing step should include a dynamic spectral mask. For each known
persistent line (P-line) of atmospheric species from the NIST database, mask
a ±δλ window around that wavelength before running NNLS. This prevents
atmospheric emission from being attributed to sample elements.

**Implementation:** Build a static mask array from NIST atmospheric P-lines
at the instrument's wavelength grid. Apply as `observed *= mask` before NNLS.
The mask width should scale with the effective resolving power.

### 8.2 Physics-Informed Validation Upgrades

**Saha-Boltzmann Cross-Validation**

The current Boltzmann consistency check (R² > 0.90 on ln(Iλ/gA) vs Eₖ)
only validates within a single ionization stage. A stronger constraint is
Saha-Boltzmann cross-validation: for elements where both stage I and II lines
are detected, verify that the Saha equation relates their populations at the
fitted (T, nₑ). If the predicted T from atomic lines differs significantly
from T predicted by ionic lines, the identification is physically inconsistent.

For fs/ps-LIBS where we expect only stages I and II, this is a powerful
discriminator — the ratio n_II/n_I must be consistent with:

    n_II/n_I = (2.41×10¹⁵ / nₑ) · T^1.5 · (U_II/U_I) · exp(-IP/kT)

Elements passing both Boltzmann linearity AND Saha consistency get a
confidence boost; those failing either get penalized. This replaces the
single R² check with a more physically grounded two-stage validation.

**Column Density Saha-Boltzmann (CD-SB) Plots**

For elements prone to self-absorption (Fe resonance lines are common in AM
alloys), the standard Boltzmann plot fails because self-absorbed line
intensities are artificially suppressed [Zhang et al. 2022, Frontiers of
Physics]. The CD-SB method bypasses the optically thin requirement by tracking
ground-state columnar densities rather than upper-state intensities. It plots:

    y_p = ln(n_I^i · l / g_I^i)  vs  x_p = E_I^i  (lower energy level)

for neutral lines, and includes an ionization correction term for ionic lines.
This yields accurate T even when resonance lines are saturated, and is
particularly valuable at later gate delays when ion recombination depletes
ionic emission [Zhang et al. 2022].

**Implementation:** Add CD-SB as an alternative T validation method in the
model selection step, falling back to it when the standard Saha-Boltzmann
R² < 0.90 (indicating potential self-absorption).

### 8.3 Advanced Retrieval and Similarity Metrics

**Spectral Entropy Similarity**

Research from the Gemini report indicates that spectral entropy similarity
outperforms over 40 alternative similarity metrics (including cosine and dot
product) for library matching tasks. It is particularly robust against noise
artifacts that generate false matches. The metric normalizes intensity arrays
to sum to 0.5, then computes the information entropy of the combined
distribution, mathematically penalizing matches that rely on stochastic noise.

**Implementation:** Replace or augment the PCA + L2 distance in the FAISS
retrieval step with entropy similarity scoring. Can be computed in the
refinement step after FAISS provides initial candidates.

**NIST Accuracy Grade Weighting**

Not all NIST transition probabilities (A_ki values) are equally reliable. NIST
assigns accuracy grades from AA (≤1% uncertainty) to E (>50% uncertainty).
The basis library should weight lines by their NIST accuracy grade so that
the L-BFGS-B optimizer prioritizes high-confidence transitions:

    w_line = {AA: 1.0, A+: 0.95, A: 0.9, B+: 0.8, B: 0.7, C+: 0.5, C: 0.3, D+: 0.2, D: 0.1, E: 0.05}

This reduces the influence of poorly-known transition probabilities on both
the NNLS decomposition and the refinement step.

**Implementation:** Add accuracy_grade column to the atomic database query.
Apply weights in the forward model's emissivity calculation.

### 8.4 Probabilistic Refinement

**Bayesian Network Expansion (BN+1)**

Instead of deterministic BIC backward elimination, model the conditional
dependencies between emission lines using a Directed Acyclic Graph (DAG).
Each emission transition is a node; edges encode the Boltzmann/Saha
relationships between lines of the same element. A candidate element is only
"accepted" if the inclusion of its full network of lines significantly
improves the posterior likelihood of the overall model.

This is more powerful than BIC because it considers the joint probability of
ALL lines of an element, not just the aggregate residual. A spurious element
with 2 coincidental matches but 5 missing expected lines would be rejected
even if the 2 matches slightly improve χ².

**Implementation:** Use `PyMC` or `numpyro` for probabilistic modeling.
The BN+1 test compares P(data | n elements) vs P(data | n+1 elements)
using Bayes factors.

**Two-Zone Plasma Model**

The design document acknowledges the risk of LTE departure in fs-LIBS.
A two-zone model (hot core + cool periphery) better captures the spatial
gradients in real laser plasmas:

    S(λ) = S_core(λ; T_core, nₑ_core) · [1 - exp(-τ_core)]
         + S_shell(λ; T_shell, nₑ_shell) · [1 - exp(-τ_shell)]

This adds 4 parameters (T_shell, nₑ_shell, R_core, R_shell) but
significantly improves relative concentration estimates for elements whose
emission originates from different plasma zones. The OPSIAL software
demonstrated that allowing T_exc ≠ T_i improved ChemCam fitting [Tan 2018].

**Implementation:** Extend the JAX forward model to optionally compute a
two-zone spectrum. Use only when single-zone residuals are poor (RSS/n > threshold).

### 8.5 Enhancement Priority Matrix

| Enhancement | Impact | Effort | Phase |
|-------------|--------|--------|-------|
| Adaptive atmospheric masking | High (precision) | Low | Phase 3 |
| NIST accuracy grade weighting | High (precision) | Low | Phase 1 |
| Saha-Boltzmann cross-validation | High (precision) | Medium | Phase 5 |
| DWT wavelet denoising | Medium (recall for trace) | Low | Phase 3 |
| CD-SB temperature validation | Medium (robustness) | Medium | Phase 5 |
| Spectral entropy similarity | Medium (precision) | Medium | Phase 2 |
| Statistical Interference Factor (SIF) | High (precision) | Medium | Phase 5 |
| ROI-based shoulder detection | Medium (recall for trace) | Medium | Phase 3 |
| Automated doublet-ratio SA correction | Medium (robustness) | Medium | Phase 4 |
| BN+1 probabilistic model selection | High (precision) | High | Future |
| Two-zone plasma model | Medium (accuracy) | High | Future |

### 8.6 Additional Enhancements (from Follow-Up Review)

**Statistical Interference Factor (SIF)**

The SIF metric (Baudelet et al.) provides a quantitative measure of how much
a specific emission line is interfered with by lines from other elements in
the matrix. For each candidate line, SIF is computed as the ratio of the
target element's predicted emissivity to the sum of all elements' emissivities
at that wavelength:

    SIF(λ) = ε_target(λ) / Σᵢ εᵢ(λ)

Lines with SIF > 0.8 are "clean" (dominated by the target element) and should
be prioritized for final element assignment. Lines with SIF < 0.3 are
heavily interfered and should be downweighted or excluded from the Boltzmann
plot and NNLS coefficient significance calculation.

**Implementation:** After NNLS decomposition identifies candidate elements,
compute SIF for each element's strongest lines using the basis spectra.
Reweight the NNLS significance test to prioritize clean lines. This is
critical for complex AM alloys (e.g., Inconel 718 where Ni/Cr/Fe/Nb lines
heavily overlap).

**ROI-Based Derivative Shoulder Detection**

For achieving >95% recall on trace elements in dense alloy spectra, the
pipeline needs to detect subtle peaks hidden in the wings of intense matrix
lines. A derivative-based shoulder detection strategy:

1. Compute 2nd derivative of the spectrum: d²I/dλ²
2. Identify shoulders as local minima in d²I/dλ² that don't correspond
   to resolved peaks but indicate inflection points
3. Use detected shoulders to seed additional Voigt profile components in
   the refinement step's forward model
4. Re-run NNLS with the shoulder positions included as additional
   constraints

This is particularly important for detecting V (0.4%) in Ti6Al4V where
V I 437.92nm sits on the wing of Ti I 438.19nm.

**Implementation:** Add shoulder detection to the preprocessing pipeline.
Feed shoulder positions into the refinement step as soft constraints
(additional Voigt components with bounded amplitudes).

**Automated Doublet-Ratio Self-Absorption Correction**

For elements with known doublet transitions from the same multiplet (e.g.,
Na D lines at 589.0/589.6nm, Ca II 393.4/396.8nm), the theoretical intensity
ratio is fixed by quantum mechanics (g_k·A_ki ratios). Any deviation from
this ratio directly quantifies the self-absorption level:

    SA = (I_observed_ratio / I_theoretical_ratio)^(1/α)

where α ≈ -0.54 is the self-absorption exponent. The true optically-thin
intensity can then be reconstructed:

    I_corrected = I_observed / SA

This is automated and calibration-free — no external references needed.
The corrected intensities should be fed back into the NNLS decomposition
and Boltzmann plot to improve both identification accuracy and T estimation.

**Implementation:** Build a lookup table of known doublet/multiplet pairs
from the NIST database. During the refinement step, check each detected
element's doublet ratios and apply SA correction where deviations > 10%
are found.

---

## 9. References

### Academic Literature
1. Hermann J, Axente E, Pelascini F, Craciun V. "Analysis of Multi-elemental
   Thin Films via Calibration-Free LIBS." Analytical Chemistry, 2019. (36 citations)
2. Tan X. "OPSIAL: A Software Package for Rigorously Calculating Optical Plasma
   Spectra and Automatically Retrieving Plasma Properties." arXiv:1802.01000, 2018.
3. Zhang N et al. "A Brief Review of Calibration-Free LIBS." Frontiers of
   Physics, 2022. (18 citations)
4. Yu Y, Yao M. "Ensemble CNNs for ChemCam Quantitative Analysis." Remote
   Sensing, 2023. (19 citations) — ECNN achieved RMSE 54% lower than PLS,
   73% lower than ELM for major element prediction.
5. Zhang C et al. "Application of Deep Learning in LIBS: A Review." AI Review,
   2023. (18 citations)
6. Dong J et al. "Genetic Algorithm for Plasma Temperature in CF-LIBS." JAAS,
   2015. (38 citations)
7. Zhao S et al. "Iterative Correction of Plasma Temperature and Spectral
   Intensity for CF-LIBS." Plasma Science & Technology, 2018.
8. Noël et al. "ALIAS: Automated Line Identification Algorithm for
   Spectroscopy." Spectrochimica Acta Part B, 2025.

### Gemini Deep Research Report
9. "Advanced Computational Strategies for Enhancing Precision and Recall in
   LIBS Pipelines" — Gemini Deep Research, March 2026. 71 websites surveyed.
   Key findings: PSO-arPLS baseline correction, Bayesian BN+1 model selection,
   spectral entropy similarity, Boltzmann linearity filtering (R² > 0.95),
   Saha-Boltzmann cross-validation for multi-stage consistency, CD-SB method
   for self-absorption robustness, NIST accuracy grade weighting, adaptive
   atmospheric masking, two-zone plasma model for spatial gradients, DWT
   wavelet denoising for single-shot SNR improvement.

### Existing Codebase
10. cflibs/manifold/ — ManifoldGenerator, SpectralEmbedder, VectorIndex
11. cflibs/radiation/spectrum_model.py — SpectrumModel forward model
12. cflibs/inversion/alias_identifier.py — Current ALIAS implementation
13. cflibs/inversion/joint_optimizer.py — JointOptimizer with softmax
14. cflibs/plasma/saha_boltzmann.py — SahaBoltzmannSolver

---

## 10. Appendix: Sizing Calculations

### A.1 Basis Library

| Parameter | Value |
|-----------|-------|
| Elements | 68 |
| Ionization stages | I, II |
| T range | 4,000 – 12,000 K |
| T steps | 50 |
| nₑ range | 1×10¹⁵ – 5×10¹⁷ cm⁻³ |
| nₑ steps | 20 |
| Grid points per element | 1,000 |
| Wavelength pixels | 4,096 |
| Total spectra | 68,000 |
| Raw storage | 68k × 4096 × 4 B = 1.1 GB |
| Compressed (gzip-4) | ~300 MB |
| PCA-30 embeddings | 68k × 30 × 4 B = 8.2 MB |

### A.2 Per-Spectrum Computation

| Step | Operations | Time (GPU) | Time (CPU) |
|------|-----------|------------|------------|
| Preprocessing | Baseline + norm | 1 ms | 5 ms |
| PCA embedding | 4096 → 30 | 0.01 ms | 0.1 ms |
| FAISS search | k=50 in 68k | 0.1 ms | 1 ms |
| Basis interpolation | 68 × bilinear | 0.1 ms | 1 ms |
| NNLS decomposition | 68 × 4096 | 1 ms | 5 ms |
| JAX refinement (10 iter) | Forward model × 10 | 50 ms | 500 ms |
| BIC pruning | ~5 re-fits | 10 ms | 100 ms |
| **Total** | | **~63 ms** | **~612 ms** |

### A.3 NNLS vs ALIAS Comparison

| Aspect | ALIAS (current) | Spectral NNLS (proposed) |
|--------|-----------------|--------------------------|
| Input | Peak positions + intensities | Full spectrum (4096 channels) |
| Information used | ~2% of spectrum | 100% of spectrum |
| Element scoring | Independent per element | Joint decomposition |
| Physical consistency | Post-hoc Boltzmann check | Built into forward model |
| Self-absorption | Not modeled | Modeled in basis spectra |
| Line overlap | Peaks compete | Naturally resolved by NNLS |
| Speed per spectrum | 28 s | ~63 ms (GPU) |
| Detection decision | CL > threshold | SNR-based + BIC |
