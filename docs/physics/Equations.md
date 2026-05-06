# Physics: Equations

This document specifies every equation CF-LIBS evaluates: the forward model
(plasma state → spectrum) and the inversion (spectrum → plasma state). It
is the canonical reference for what the code computes and how each
quantity is defined.

For the **assumptions** these equations rest on (LTE, optical thinness,
single-zone uniformity, …) and the **regime of validity** of each, see
[Assumptions and Validity](Assumptions_And_Validity.md).

For derivations and citations, the references are listed at the end.

---

## Notation and Units

| Symbol | Meaning | Unit (default in code) |
|--------|---------|-------------------------|
| `T`, `T_e` | Electron / excitation temperature | K (also exposed as eV) |
| `T_g` | Gas / heavy-particle temperature | K (defaults to `T_e`) |
| `n_e` | Electron number density | cm⁻³ |
| `n_s` | Number density of species `s` | cm⁻³ |
| `n_z`, `n_z+1` | Number density of ionization stage `z` and `z+1` | cm⁻³ |
| `C_s` | Mass fraction of element `s` | dimensionless |
| `λ`, `λ₀` | Wavelength (running, line center) | nm (vacuum) |
| `A_ki` | Einstein spontaneous-emission coefficient | s⁻¹ |
| `g_k`, `g_i` | Statistical weights of upper / lower level | dimensionless |
| `E_k`, `E_i` | Upper / lower level energy | eV |
| `χ_z` | Ionization potential of stage `z` | eV |
| `U_s(T)` | Internal partition function of species `s` | dimensionless |
| `L` | Plasma path length along line of sight | m |
| `F` | Experimental factor (geometry × detector × etc.) | system-dependent |
| `k`, `k_B` | Boltzmann constant | 8.617333…×10⁻⁵ eV/K |
| `h` | Planck constant | 6.62607015×10⁻³⁴ J·s |
| `c` | Speed of light | 2.99792458×10⁸ m/s |
| `m_e` | Electron mass | 9.1093837×10⁻³¹ kg |

`cflibs.core.constants` is the authoritative source; the values above are
informational.

---

**Forward model.** The forward model maps a plasma state `(T, n_e, {C_s})`
to a synthetic spectrum `I(λ)`. The pipeline is:

```
PlasmaState → Saha–Boltzmann → level populations
            → line emissivity → ε(λ)
            → optically thin transport → I_continuum-free(λ)
            → instrument convolution → I(λ)
```

(physics-boltzmann-distribution)=
## 1. Boltzmann Distribution (Excitation)

Within an ionization stage, the population of level `k` is

$$
\frac{n_k}{n_s^{(z)}} = \frac{g_k}{U_s^{(z)}(T)} \exp\!\left(-\frac{E_k}{k_B T}\right)
$$

where `n_s^{(z)}` is the total density of species `s` in ionization stage
`z`.

Implemented in `cflibs.plasma.saha_boltzmann.SahaBoltzmannSolver.solve_level_population`.

(physics-saha-equation)=
## 2. Saha Equation (Ionization Balance)

Between two adjacent ionization stages,

$$
\frac{n_{z+1}\, n_e}{n_z}
=
\frac{2\, U_{z+1}(T)}{U_z(T)}
\left(\frac{2\pi m_e k_B T}{h^{2}}\right)^{3/2}
\exp\!\left(-\frac{\chi_z}{k_B T}\right).
$$

In code (`cflibs/plasma/saha_boltzmann.py`) this is rearranged with the
constant `SAHA_CONST_CM3` to operate in cm⁻³, eV, K. The factor of 2 is
the electron statistical weight.

For elements with multiple stages, the `solve_ionization_balance` routine
solves the recursion `n_{z+1}/n_z` for `z = 0, 1, 2, …` constrained by

$$
\sum_{z} n_z = n_s,
$$

where `n_s` is the total number density of element `s`.

(physics-partition-function)=
## 3. Partition Function (Irwin Polynomial)

CF-LIBS uses the Irwin (1981) polynomial form

$$
\log_{10} U_s^{(z)}(T) = \sum_{n=0}^{N} a_n^{(s,z)} \big(\log_{10} T\big)^n,
$$

with coefficients `a_n` stored per `(element, ionization_stage)` in the
`partition_functions` table. When coefficients are unavailable for a
given species, the solver falls back to a direct sum over energy levels:

$$
U_s^{(z)}(T) = \sum_{k} g_k \exp\!\left(-\frac{E_k}{k_B T}\right).
$$

Implemented in `SahaBoltzmannSolver.calculate_partition_function`.

(physics-line-emissivity)=
## 4. Line Emissivity

For a single transition `k → i` with upper-level population `n_k`,

$$
\varepsilon_{ki}(\lambda)
=
\frac{h c}{4\pi\, \lambda}\, A_{ki}\, n_k\, \phi_{ki}(\lambda)
\quad\big[\text{W m}^{-3}\,\text{nm}^{-1}\,\text{sr}^{-1}\big],
$$

where `φ_ki(λ)` is the area-normalized line profile (∫ φ dλ = 1).
Implemented in `cflibs.radiation.emissivity.calculate_line_emissivity`.

The total emissivity at wavelength `λ` is the sum over all transitions
included in the model:

$$
\varepsilon(\lambda) = \sum_{ki} \varepsilon_{ki}(\lambda).
$$

(physics-line-profiles)=
## 5. Line Profiles

Three profiles are available in `cflibs.radiation.profiles`:

**Doppler / Gaussian** (thermal motion of emitters of mass `m`):

$$
\sigma_D = \lambda_0 \sqrt{\frac{k_B T}{m c^{2}}},
\qquad
G(\lambda) = \frac{1}{\sigma_D \sqrt{2\pi}} \exp\!\left[-\frac{(\lambda-\lambda_0)^2}{2\sigma_D^{2}}\right].
$$

**Stark / Lorentzian** (electron impacts):

$$
\gamma_{\text{Stark}}(n_e, T)
=
w_{\text{ref}}
\left(\frac{n_e}{n_{e,\text{ref}}}\right)
\left(\frac{T}{T_{\text{ref}}}\right)^{-\alpha},
$$

with `w_ref`, `α` per-line in the atomic database (or estimated from the
charge-state semi-empirical model when missing). Implemented in
`cflibs.radiation.stark`.

**Voigt** (convolution of Gaussian and Lorentzian):

$$
V(\lambda; \sigma, \gamma)
=
\frac{\operatorname{Re}\!\big[w(z)\big]}{\sigma\sqrt{2\pi}},
\qquad
z = \frac{(\lambda-\lambda_0) + i\gamma}{\sigma\sqrt{2}},
$$

where `w(z)` is the Faddeeva function. The CPU implementation uses the
Humlicek W4 algorithm (relative error <10⁻⁴); the JAX implementation
(`voigt_profile_jax`) uses the Weideman 32-term rational approximation,
which is gradient-stable for MCMC and L-BFGS.

(physics-optically-thin-transport)=
## 6. Optically Thin Transport

For an optically thin plasma of uniform path length `L`, the spectral
intensity emerging along the line of sight is

$$
I(\lambda) = \varepsilon(\lambda)\, L.
$$

This is the default. When self-absorption is non-negligible it must be
corrected per line; see [Self-Absorption](#physics-self-absorption-and-curve-of-growth)
below.

(physics-instrument-convolution)=
## 7. Instrument Convolution

The spectrometer broadens every emission feature by its line-spread
function. CF-LIBS approximates this as a Gaussian, with two modes:

**Fixed FWHM** (standard / monochromator):

$$
\sigma_{\text{inst}} = \frac{\text{FWHM}}{2\sqrt{2 \ln 2}}.
$$

**Resolving-power mode** (echelle): if `R = λ/Δλ` is set,

$$
\text{FWHM}(\lambda) = \lambda / R,
\qquad
\sigma_{\text{inst}}(\lambda) = \frac{\lambda}{R\, 2\sqrt{2 \ln 2}}.
$$

The convolution

$$
I_{\text{measured}}(\lambda)
=
\big(I \ast G_{\sigma_{\text{inst}}}\big)(\lambda)
\cdot R_{\text{det}}(\lambda)
$$

multiplies in any optional spectral response `R_det(λ)`. Implemented in
`cflibs.instrument.convolution.apply_instrument_function`.

The Doppler/Stark width inside `φ_ki(λ)` and the instrument width are
combined in quadrature in the Gaussian channel, producing a final Voigt
with effective `σ² = σ_D² + σ_inst²` and Lorentzian wing `γ_Stark`.

---

**Inversion.** The iterative CF-LIBS solver
(`cflibs.inversion.solver.IterativeCFLIBSSolver`) inverts the forward
model. The algorithm is standard Ciucci/Tognoni CF-LIBS with refinements
documented below.

(1-boltzmann-plot)=
(physics-boltzmann-plot)=
## 1. Boltzmann Plot

For an LTE plasma, the integrated intensity of line `k → i` is

$$
I_{ki}
=
F\, C_s\, \frac{g_k\, A_{ki}}{U_s(T)}
\exp\!\left(-\frac{E_k}{k_B T}\right)\, \frac{1}{\lambda_{ki}}.
$$

Taking the natural log and rearranging:

$$
\underbrace{\ln\!\left(\frac{I_{ki}\, \lambda_{ki}}{g_k\, A_{ki}}\right)}_{y}
=
-\underbrace{\frac{1}{k_B T}}_{\text{slope}}\, E_k
+
\underbrace{\ln\!\left(\frac{F\, C_s}{U_s(T)}\right)}_{q_s\ \text{(intercept)}}.
$$

Plotting `y` against `E_k` for a single species:

- **Slope** = `−1 / (k_B T)` → `T`.
- **Intercept** `q_s` is element-specific and proportional to `ln(F C_s)`
  modulo `U_s(T)`.

CF-LIBS fits the Boltzmann plot per element. With multiple elements, the
solver enforces a **common slope** (all elements share the same plasma
temperature) while letting each element have its own intercept. This is
the multi-element common-slope fit, implemented in
`cflibs.inversion.physics.boltzmann_fit`.

Available fitting estimators (`FitMethod`):

- `WEIGHTED_LS` — ordinary weighted least squares with intensity-error
  weights `w_i = 1/σ_i²`.
- `SIGMA_CLIP` — iterative re-fit dropping `>3σ` outliers (default).
- `RANSAC` — random sample consensus for heavy contamination.
- `HUBER` — Huber M-estimator for medium-tail noise.

(physics-saha-correction)=
## 2. Saha Correction (Map Ionic Lines onto the Neutral Plane)

If only the neutral lines were used, ionic intensities would be discarded
and the lever arm in `E_k` would shrink. Instead, every ionic-line
intercept `q_s^{(z)}` is brought back to the neutral plane via Saha:

$$
q_s^{(0)} = q_s^{(z)} - z \ln S(T, n_e),
$$

where `S(T, n_e)` is the Saha factor for stage transitions:

$$
S(T, n_e)
=
\frac{1}{n_e}
\left(\frac{2\pi m_e k_B T}{h^{2}}\right)^{3/2}
\exp\!\left(-\frac{\chi_{z-1}}{k_B T}\right)
\frac{2 U_z(T)}{U_{z-1}(T)}.
$$

Implemented as `IterativeCFLIBSSolver._saha_correction`. After this
correction every line — neutral or ionic — sits on the same Boltzmann
plot for its element.

(closure-equation)=
(3-closure-equation)=
(physics-closure-equation)=
## 3. Closure Equation

The intercepts `q_s` of each element constrain `F C_s / U_s(T)`. Closure
turns these into mass fractions by enforcing a normalization condition.

**Standard mode** (`closure_mode: standard`):

$$
\sum_s C_s = 1
\quad\Rightarrow\quad
F = \sum_s U_s(T)\, e^{q_s},
\qquad
C_s = \frac{U_s(T)\, e^{q_s}}{F}.
$$

**Matrix mode** (`closure_mode: matrix`): if the matrix element `m` has a
known mass fraction `C_m`,

$$
F = \frac{U_m(T)\, e^{q_m}}{C_m},
\qquad
C_s = \frac{U_s(T)\, e^{q_s}}{F}.
$$

Useful for steel (Fe is matrix), brass (Cu is matrix), Ti alloys (Ti is
matrix), etc.

**Oxide mode** (`closure_mode: oxide`): for geological samples in which
elements are reported as their stable oxides (SiO₂, Al₂O₃, Fe₂O₃, …),
each element fraction `C_s` is converted to the corresponding oxide mass
fraction using the stoichiometric ratio

$$
C_{s\text{-oxide}}
=
C_s \cdot \frac{M_{s\text{-oxide}}}{n_s\, M_s},
$$

and oxides are normalized so that `Σ C_oxide = 1`.

The closure code is in `cflibs.inversion.physics.closure`.

(physics-ne-update)=
## 4. Self-Consistent Update of `n_e`

Saha couples `T` and `n_e`. After applying closure, the solver re-derives
`n_e` from the inferred populations using the Saha equation again
(consistency between observed `q_s` of neutral and ionic lines, plus
charge balance `n_e = Σ_s C_s n_s × ⟨z_s⟩`). The new `n_e` feeds back
into the Saha correction in step 2.

(physics-convergence)=
## 5. Convergence

Iterate steps 1–4 until both

$$
\frac{|T^{(k+1)} - T^{(k)}|}{T^{(k)}} < \tau_T,
\qquad
\frac{|n_e^{(k+1)} - n_e^{(k)}|}{n_e^{(k)}} < \tau_{n_e},
$$

where `τ_T` and `τ_n_e` are `t_tolerance_k` and `ne_tolerance_frac` from
the analysis config. Default `max_iterations = 20`.

---

(self-absorption-and-curve-of-growth)=
(physics-self-absorption-and-curve-of-growth)=
## Self-Absorption and Curve of Growth

For an optically thin line, `I_obs ∝ n L`. As optical depth `τ` rises,
the curve of growth bends and the linear regime fails:

$$
\tau_0
=
\frac{\pi e^2}{m_e c}\, f_{ki}\, n_i\, L\, \frac{1}{\Delta\nu_D},
\qquad
\frac{I_{\text{obs}}}{I_{\text{thin}}}
=
\frac{1 - e^{-\tau_0}}{\tau_0}
\quad\text{(Doppler core, no wings).}
$$

CF-LIBS implements two independent self-absorption corrections:

- **Iterative `SelfAbsorptionCorrector`** — estimates `τ` per line from
  the lower-level population (which itself depends on the not-yet-known
  composition), applies the curve-of-growth correction, and iterates.
  Lines exceeding `max_optical_depth` are masked rather than corrected.
- **CDSB (Column-Density Self-Absorption Correction)** — uses the
  full Voigt curve of growth for stronger self-absorption regimes.
  Implemented in `cflibs.inversion.physics.cdsb`.

Both methods rely on knowing the absolute column density `n_i L`, which
in turn requires an absolute intensity calibration or a known-fraction
matrix line as a reference. Without absolute calibration, exclude
resonance and other strong/saturable lines (`exclude_resonance: true`)
rather than attempting correction.

---

## LTE Validity: McWhirter Criterion

LTE in a homogeneous optically thin plasma requires that collisional
de-excitation rates dominate radiative ones. The standard sufficiency
criterion is

$$
n_e \;\ge\; 1.6 \times 10^{12}\; \sqrt{T}\; (\Delta E)^{3} \quad \text{cm}^{-3},
$$

with `T` in K and `ΔE` in eV the largest energy gap encountered in the
levels feeding the lines used. The solver evaluates this at convergence
and reports `lte_mcwhirter_satisfied` and `lte_n_e_ratio` in
`result.quality_metrics`. A violation invalidates the inferred `T` and
composition even if the math converged.

For the deeper conditions and how to act on a McWhirter violation, see
[Assumptions and Validity](Assumptions_And_Validity.md).

---

(bayesian-forward-model)=
(physics-bayesian-forward-model)=
## Bayesian Forward Model

The Bayesian forward model
(`cflibs.inversion.solve.bayesian.BayesianForwardModel`) wraps the same
physics in JAX-traced form for MCMC and gradient-based fitting. Likelihood
is

$$
\ln p(\mathbf{I}_{\text{obs}} \mid T, n_e, \mathbf{C})
=
- \frac{1}{2} \sum_j \frac{(I_{\text{obs},j} - I_{\text{model},j})^2}{\sigma_j^2}
- \sum_j \ln \sqrt{2\pi \sigma_j^2}.
$$

Three noise models are exposed:

- `gaussian` — `σ_j² = σ_readout²`, constant.
- `poisson` — `σ_j² = I_j`, shot-noise dominated.
- `combined` — `σ_j² = σ_readout² + α I_j + dark`, realistic CCD model.

Standard priors (`PriorConfig`):

- `T ~ Uniform(T_min_eV, T_max_eV)` or `LogUniform(...)`.
- `n_e ~ LogUniform(n_e_min, n_e_max)` (Jeffreys prior, scale-invariant).
- `{C_s} ~ Dirichlet(α)` so that `Σ C_s = 1` is enforced by construction.

Sampling uses NumPyro NUTS by default; nested sampling via dynesty is
available for evidence calculation and model comparison.

---

(references)=
(physics-references)=
## References

The numbered references below are cited inline by `[n]`.

1. Ciucci, A., et al. *New procedure for quantitative elemental analysis
   by laser-induced plasma spectroscopy.* Applied Spectroscopy 53.8
   (1999): 960–964.
2. Tognoni, E., et al. *Calibration-free laser-induced breakdown
   spectroscopy: state of the art.* Spectrochimica Acta B 65.1 (2010):
   1–14.
3. Aragón, C., and J. A. Aguilera. *Characterization of laser induced
   plasmas by optical emission spectroscopy: A review of experiments and
   methods.* Spectrochimica Acta B 63.9 (2008): 893–916.
4. Irwin, A. W. *Polynomial partition function approximations of 344
   atomic and molecular species.* Astrophys. J. Suppl. 45 (1981):
   621–633.
5. Griem, H. R. *Spectral Line Broadening by Plasmas.* Academic Press
   (1974).
6. Humlicek, J. *Optimized computation of the Voigt and complex
   probability functions.* JQSRT 27.4 (1982): 437–444.
7. Salzmann, D. *Atomic Physics in Hot Plasmas.* Oxford University Press
   (1998).
8. McWhirter, R. W. P. In *Plasma Diagnostic Techniques*, eds. Huddlestone
   & Leonard (Academic Press, 1965), Ch. 5.
9. Cristoforetti, G., et al. *Local thermodynamic equilibrium in
   laser-induced breakdown spectroscopy: beyond the McWhirter criterion.*
   Spectrochimica Acta B 65.1 (2010): 86–95.
10. Hou, Z., et al. *A comprehensive review on calibration-free
    laser-induced breakdown spectroscopy.* Front. Phys. 16 (2021).

---

## See Also

- [Assumptions and Validity](Assumptions_And_Validity.md) — what each
  equation assumes about your plasma and when it fails.
- [Inversion Algorithm](Inversion_Algorithm.md) — line-by-line walkthrough
  of the iterative CF-LIBS algorithm including pseudocode.
- [Codebase Architecture](../reference/Codebase_Architecture.md) — where
  each equation lives in the source tree.
- [API Reference](../reference/API_Reference.md) — call signatures.
