# Bayesian and Optimal-Estimation Retrieval for CF-LIBS Plasma Spectroscopy

**Topic:** bayesian-oe — Bayesian / optimal-estimation (Rodgers) retrieval of plasma parameters
and composition from spectra; NUTS/HMC for spectral fitting; Dirichlet / simplex priors for
composition; Poisson + readout noise models for CCD/ICCD detectors.

---

## 1. Governing Equations — Canonical Forms

### 1.1 The Bayesian Retrieval Problem

Given measured spectrum **y** (N-vector, counts or intensity), find the posterior over state
vector **x** = (T, nₑ, C₁, …, Cₛ, …):

```
p(x | y) ∝ p(y | x) · p(x)          [Bayes' theorem]
```

The state vector for CF-LIBS is typically:

```
x = { T [K],  nₑ [cm⁻³],  C_s [mol. fraction, s=1..S],
      possible nuisance: instrumental width Δλ, baseline coefficients b_k }
```

Constraints: all Cₛ ≥ 0, ΣCₛ = 1 (composition lives on the S-1 simplex).

---

### 1.2 Rodgers Optimal Estimation (OE) — MAP / Gauss–Newton Framework

**Reference:** Rodgers, C. D. (2000). *Inverse Methods for Atmospheric Sounding: Theory and
Practice.* World Scientific. (The canonical reference for OE retrievals.)

The **forward model** maps state to spectrum:

```
y = F(x) + ε,    ε ~ N(0, Sₑ)
```

where Sₑ is the measurement-error covariance (N×N).

**A priori:** x ~ N(xₐ, Sₐ), where xₐ is the prior mean and Sₐ is prior covariance.

**Cost function (MAP for Gaussian prior + Gaussian noise):**

```
J(x) = ½ (y − F(x))ᵀ Sₑ⁻¹ (y − F(x))  +  ½ (x − xₐ)ᵀ Sₐ⁻¹ (x − xₐ)
```

**Gauss–Newton iteration** (linearised update, Rodgers eq. 5.8):

```
xᵢ₊₁ = xₐ + Sₐ Kᵢᵀ (Kᵢ Sₐ Kᵢᵀ + Sₑ)⁻¹ (y − F(xᵢ) + Kᵢ(xᵢ − xₐ))
```

where **K** = ∂F/∂x|_{xᵢ} is the Jacobian (N×M), evaluated at current iterate.

Equivalent form (avoids large matrix inverse for M ≪ N):

```
xᵢ₊₁ = xₐ + (Kᵢᵀ Sₑ⁻¹ Kᵢ + Sₐ⁻¹)⁻¹ Kᵢᵀ Sₑ⁻¹ (y − F(xᵢ) + Kᵢ(xᵢ − xₐ))
```

**Posterior covariance** at convergence (linear approximation):

```
Ŝ = (Kᵀ Sₑ⁻¹ K + Sₐ⁻¹)⁻¹          [Rodgers eq. 5.9]
```

**Averaging kernel** (how well the retrieval resolves information):

```
A = Ŝ Kᵀ Sₑ⁻¹ K = I − Ŝ Sₐ⁻¹      [Rodgers eq. 5.10]
```

Degrees of freedom for signal: d_s = tr(A).

**Symbol conventions (Rodgers standard):**

| Symbol | Meaning | Units |
|--------|---------|-------|
| y | measured spectrum | [counts or W m⁻² sr⁻¹ nm⁻¹] |
| x | state vector | various |
| xₐ | prior/a priori state | same as x |
| F(x) | forward model | same as y |
| K | Jacobian ∂F/∂x | [y-units / x-units] |
| Sₑ | measurement error covariance | [y-units²] |
| Sₐ | a priori covariance | [x-units²] |
| Ŝ | posterior covariance | [x-units²] |
| A | averaging kernel matrix | dimensionless |

**Common factor-of-2 pitfall:** J has the ½ prefactor; the Hessian of J is (KᵀSₑ⁻¹K + Sₐ⁻¹),
not 2× that. Dropping the ½ shifts the MAP but not the Gauss–Newton convergence direction;
it does shift the uncertainty estimate Ŝ by a factor of 2 if naively inverted from the
(un-halved) Hessian.

---

### 1.3 Full Bayesian Posterior — MCMC / NUTS / HMC

When the forward model F is nonlinear and the Gaussian linearisation is inaccurate (common
for CF-LIBS where T enters exponentially via Boltzmann factor), the full posterior must be
sampled:

```
log p(x | y) = log p(y | x) + log p(x) + const
```

For Gaussian noise likelihood + log-normal or normal prior:

```
log p(y | x) = −½ (y − F(x))ᵀ Sₑ⁻¹ (y − F(x)) − ½ log det(2π Sₑ)
```

**Hamiltonian Monte Carlo (HMC):**  
Augment state with momentum p ~ N(0, M) (mass matrix M). Hamiltonian:

```
H(x, p) = −log p(x | y) + ½ pᵀ M⁻¹ p
```

Leapfrog integration of Hamilton's equations for L steps of size ε. The gradient
∇_x log p(x|y) = −∇_x J(x) must be computed; JAX autodiff provides this for free.

**NUTS (No-U-Turn Sampler)** (Hoffman & Gelman, 2014):  
Eliminates the L hyperparameter by recursively doubling the trajectory until a "U-turn"
condition is triggered. Uses slice sampling within the candidate tree for reversibility.
U-turn condition: (θ⁺ − θ⁻) · r⁻ < 0  OR  (θ⁺ − θ⁻) · r⁺ < 0, where r±  are the
endpoint momenta.

**Dual averaging (Nesterov) step-size adaptation** during warm-up automatically tunes ε to
target acceptance probability δ ≈ 0.65–0.85.

---

### 1.4 Detector Noise Models

#### 1.4.1 Ideal Poisson (shot-noise limited)

```
yₖ ~ Poisson(λₖ)    where λₖ = g · Iₖ(x) · t_exp + b_dark
```

- g = detector gain [counts/photon or ADU/electron]
- Iₖ(x) = model photon flux at pixel k
- t_exp = exposure / gate time
- b_dark = dark current contribution

Log-likelihood:

```
log p(y | x) = Σₖ [ yₖ log λₖ(x) − λₖ(x) − log(yₖ!) ]
```

For high counts (yₖ ≫ 1) Poisson → Gaussian with σₖ² = λₖ (Anscombe approximation also used).

#### 1.4.2 Poisson + Gaussian Readout (standard CCD/ICCD)

Each pixel k is independently:

```
yₖ = Poisson(λₖ) convolved with N(0, σ_RON²)
```

Exact likelihood requires sum-over-hidden-photon-counts; practically approximated as:

```
yₖ ~ N( λₖ,  λₖ + σ_RON² )     [Poisson variance + readout variance]
```

This is valid when λₖ is not near zero. Likelihood:

```
log p(y | x) = −½ Σₖ [(yₖ − λₖ(x))² / (λₖ(x) + σ_RON²) + log(λₖ(x) + σ_RON²)]
```

Note: the denominator is *pixel-dependent* and must be re-evaluated at each MCMC step when
using this signal-dependent variance. Omitting this (fixing σ² to its mean) biases the
uncertainty estimates.

**ICCD-specific notes:**
- Intensifier gain G ≫ 1 multiplies photon shot noise: effective variance ≈ G² λₖ/QE
- Clock-induced charge (CIC) and gating transients add non-Poissonian backgrounds
- Dark frame subtraction before inference shifts the Poisson mean to a possibly negative value;
  truncated Poisson or background subtraction needs care in the likelihood.

#### 1.4.3 Simplified Homoscedastic Gaussian (common in plasma Bayesian literature)

Many plasma spectroscopy papers (e.g., Kasim et al. 2019, Bowman et al. 2024) simply use:

```
yₖ ~ N( Fₖ(x),  σ² )      with σ common or per-region
```

This is accurate only in the high-count, readout-dominated regime. It is the same as
minimising χ² = Σ(yₖ − Fₖ)²/σ² with uniform σ. Fine for T/nₑ estimation from strong lines;
inaccurate for faint lines or Poisson-limited LIBS transients.

---

### 1.5 Composition Priors — Simplex Constraints

#### 1.5.1 Dirichlet Prior (conjugate for Multinomial)

```
C = (C₁, …, Cₛ) ~ Dir(α₁, …, αₛ)
```

PDF on the S-1 simplex:

```
p(C | α) = Γ(Σαₛ) / [Π Γ(αₛ)] · Π Cₛ^(αₛ−1)
```

- Symmetric uniform: αₛ = 1 for all s → flat on the simplex
- Sparse prior: αₛ = ½ → Jeffreys prior for multinomials, favours corners
- Informative: αₛ = α₀ Cₛ^(prior) where α₀ is precision

**Pitfall:** The Dirichlet forces sub-compositional independence (neutral compositions,
Aitchison 1986). It cannot model positive correlations between elements that co-occur (e.g.,
Fe/Ni in steel).

#### 1.5.2 Logistic-Normal / ILR Prior (Aitchison geometry)

More flexible: apply isometric log-ratio (ILR) transform to map simplex → ℝˢ⁻¹, then place
multivariate normal prior:

```
η = ILR(C) ~ N(μ_η, Σ_η)
C = ILR⁻¹(η)   (softmax-like inversion through successive binary partitions)
```

ILR transform (Egozcue et al. 2003): for a sequential binary partition with basis ψₖ:

```
ηₖ = √(rₖ⁺ rₖ⁻ / (rₖ⁺ + rₖ⁻)) · log( geometric_mean(C_in_ₖ) / geometric_mean(C_out_ₖ) )
```

where rₖ⁺ and rₖ⁻ are the numbers of components in each partition at level k.

In practice for MCMC, the additive log-ratio (ALR) is simpler:

```
ηₛ = log(Cₛ / Cₛ₀)  for s = 1, …, S−1  (ₛ₀ is reference element)
Cₛ = exp(ηₛ) / (1 + Σ exp(ηₛ))
Cₛ₀ = 1 / (1 + Σ exp(ηₛ))
```

MCMC runs unconstrained over η ∈ ℝˢ⁻¹; closure is automatic.

**NumPyro pattern:**

```python
# ALR reparameterization — unconstrained MCMC, simplex closure guaranteed
eta = numpyro.sample("eta", dist.Normal(0, 2).expand([S-1]))   # log-ratios
C_raw = jnp.concatenate([jnp.exp(eta), jnp.array([1.0])])
C = C_raw / C_raw.sum()   # softmax closure
```

Or using NumPyro's built-in Dirichlet:

```python
C = numpyro.sample("C", dist.Dirichlet(alpha * jnp.ones(S)))
```

---

## 2. Common Implementation Pitfalls and Correct Treatment

### 2.1 Exponential Sensitivity in the Boltzmann/Saha Forward Model

The emission intensity is:

```
Iₗ ∝ Cₛ · Aₖᵢ gₖ / U(T) · exp(−Eₖ / kT) · (transition-specific factors)
```

The ∂I/∂T gradient is ∝ Eₖ/(kT²), which varies by a factor of ~10 across the UV-NIR range
and becomes very large at low T. This means:

- The posterior is highly non-Gaussian in T; NUTS is much safer than Laplace / OE.
- HMC step size ε must account for the varying scale: use a dense mass matrix or diagonal
  M⁻¹ = diag(Var(T), Var(nₑ), Var(C₁), …).
- ALR-transformed concentrations remove the 0–1 hard boundary but do not remove scale
  mismatch: T ~ 10⁴ K while η ~ 0±2; use separate mass matrix entries.

### 2.2 Degeneracy Between T and Cₛ at Fixed F(x)

In CF-LIBS, lowering T can be compensated by raising the total number density (i.e., plasma
depth L or ΣCₛ nₑ). This degeneracy is only broken by including multiple ionisation stages
(Saha), which gives a distinct T dependence. If only neutral lines are used, the posterior is
a ridge; NUTS will explore it but mixing will be slow.

**Fix:** Include at least one ionic line per element when available; parameterize (T, nₑ)
jointly; set informative prior on nₑ from Stark width measurement.

### 2.3 Closure Constraint Handling

Three correct approaches (in order of MCMC efficiency):

1. **ALR / ILR reparameterization** (recommended): unconstrained η ∈ ℝˢ⁻¹; no boundary
   issues; best mixing for NUTS.
2. **Dirichlet with concentration = 1**: flat on simplex; NUTS must remain inside simplex
   (NumPyro handles this automatically via `dist.Dirichlet`).
3. **Softmax of unconstrained logits**: equivalent to ALR with reference = log-sum-exp
   normaliser; slightly less numerically stable but common in deep-learning frameworks.

**Do NOT:** constrain one element as `Cₛ₀ = 1 − ΣCₛ` and place independent uniform [0,1]
priors on each other element. This creates incorrect geometry (non-uniform on simplex),
truncation artefacts, and poor HMC mixing.

### 2.4 Signal-Dependent vs. Fixed Noise Variance

Using a fixed σ in a Poisson-limited regime biases the fit toward strong lines (which have
smaller relative noise) and over-weights faint lines (which have larger relative noise).

Correct Poisson-Gaussian likelihood weight:

```
wₖ = 1 / (λₖ(x) + σ_RON²)    [heteroscedastic, must update at each step]
```

This is more expensive but necessary when SNR < 100 on any line used in the fit.

### 2.5 Log-Likelihood Normalisation Constants

For the Gaussian likelihood with signal-dependent variance:

```
log p(y|x) = −½ Σₖ [(yₖ − λₖ)²/(λₖ + σ_RON²) + log(λₖ + σ_RON²)] − (N/2)log(2π)
```

The `log(λₖ + σ_RON²)` term is often dropped in LS fitting but matters for model comparison
(WAIC, LOO-CV) and for evidence estimation. Include it.

### 2.6 Prior on Temperature: Log-Scale vs. Linear Scale

T enters the Boltzmann exponential; a uniform prior on log(T) (i.e., Jeffreys-style) is
typically more appropriate than uniform on T, since information about T is extracted from
line ratios (intrinsically log-scale quantities). In practice:

```
log T ~ N(log T_prior, σ_logT)   with σ_logT ≈ 0.2–0.5
```

### 2.7 NUTS Warm-Up / Step-Size Tuning

- Target acceptance probability δ = 0.65–0.85 for NUTS (higher δ → shorter steps → slower
  but more reliable; Betancourt 2017 recommends δ ≈ 0.8).
- Warm-up should be ≥ 500 steps for plasma retrievals with 5–15 parameters; use 1000 if
  the model includes > 10 elements.
- Check: effective sample size (ESS) per chain > 100, R̂ < 1.01 for all parameters.
- If R̂ diverges or ESS is tiny, suspect: (a) label switching in multi-zone model, (b)
  strong T–nₑ degeneracy, (c) inadequate mass matrix adaptation.

### 2.8 Rodgers OE — When to Use vs. Full MCMC

OE (Gauss–Newton) is appropriate when:
- The forward model is weakly nonlinear in x (Jacobian K approximately constant).
- Gaussian noise is an adequate approximation.
- Speed is critical (OE converges in 3–5 iterations vs. thousands of MCMC steps).

Use MCMC when:
- Strong nonlinearity (T enters exponentially; Saha has T^(5/2) / exp(χ/kT)).
- Non-Gaussian posterior (multi-modal, ridge-shaped due to T/nₑ degeneracy).
- Uncertainty is of primary interest (OE posterior covariance is only a linearised
  approximation).
- Model comparison via log marginal likelihood is needed.

For CF-LIBS: MCMC (NUTS via NumPyro) is the rigorous choice for final uncertainty
quantification; OE (or the classical iterative CF-LIBS loop) can be used as a fast
first-pass initialiser.

---

## 3. Key References

1. **Rodgers, C. D. (2000).** *Inverse Methods for Atmospheric Sounding: Theory and Practice.*
   World Scientific, Singapore. ISBN 981-02-2740-X.
   — The canonical MAP / OE reference. Equations 5.8–5.10 define the Gauss–Newton iterate,
   posterior covariance, and averaging kernel in the standard notation used by all atmospheric
   retrieval codes.

2. **Hoffman, M. D. & Gelman, A. (2014).** The No-U-Turn Sampler: Adaptively Setting Path
   Lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(47),
   1593–1623. https://jmlr.org/papers/v15/hoffman14a.html
   — The NUTS paper. Defines the recursive doubling tree, U-turn stopping criterion, and
   dual-averaging step-size adaptation. The standard algorithm implemented in NumPyro,
   Stan, and PyMC.

3. **Neal, R. M. (2011).** MCMC Using Hamiltonian Dynamics. In *Handbook of Markov Chain
   Monte Carlo* (Brooks, Gelman, Jones, Meng, eds.), Chapter 5, pp. 113–162. CRC Press.
   arXiv:1206.1901.
   — Comprehensive derivation of HMC leapfrog integration, mass matrix role, step-size
   sensitivity, and connection to molecular dynamics. Essential background for NUTS.

4. **Phan, D., Pradhan, N. & Jankowiak, M. (2019).** Composable Effects for Flexible and
   Accelerated Probabilistic Programming in NumPyro. arXiv:1912.11554.
   https://arxiv.org/abs/1912.11554
   — NumPyro paper. Describes the JIT-compiled iterative NUTS implementation, effect
   handlers, and JAX integration. The practical reference for CF-LIBS Bayesian code using
   NumPyro + JAX.

5. **Aitchison, J. (1982).** The statistical analysis of compositional data (with discussion).
   *Journal of the Royal Statistical Society Series B*, 44(2), 139–177.
   DOI: 10.1111/j.2517-6161.1982.tb01195.x
   — Introduces log-ratio transforms for compositional data; establishes that the simplex
   has a natural Euclidean structure in log-ratio coordinates. Foundation for ALR/ILR
   reparameterization of composition priors in MCMC.

6. **Egozcue, J. J., Pawlowsky-Glahn, V., Mateu-Figueras, G. & Barceló-Vidal, C. (2003).**
   Isometric Logratio Transformations for Compositional Data Analysis.
   *Mathematical Geology*, 35(3), 279–300. DOI: 10.1023/A:1023818214614
   — Defines the ILR transform (the isometric version of the log-ratio family) preserving
   Aitchison distances. Used for ILR-based Bayesian priors that are orthonormal on the
   simplex.

7. **Bowman, C., Mildner, J. M., Pasley, J. & Wilson, S. R. (2024).** Automated Bayesian
   high-throughput estimation of plasma temperature and density from emission spectroscopy.
   *Review of Scientific Instruments*, 95, 073520.
   DOI: 10.1063/5.0190924. arXiv:2312.12674
   — Demonstrates Bayesian framework (Gaussian likelihood, grid-based posterior) for plasma
   T and nₑ from emission spectroscopy; addresses systematic line-parameter uncertainties
   explicitly via Gaussian-process likelihood.

8. **Kruger, S. E. (2024).** Thinking Bayesian for plasma physicists.
   *Physics of Plasmas*, 31, 050901. DOI: 10.1063/5.0200608
   — Tutorial review of Bayesian methods written for the plasma physics community. Covers
   prior specification, likelihood construction, Gaussian process regression, and inverse
   problem framing; good entry point.

9. **Kasim, M. F. et al. (2019).** Quantitative single shot and spatially resolved plasma
   fluorescence spectra with a Bayesian approach. *Physical Review E*, 100, 033302.
   DOI: 10.1103/PhysRevE.100.033302
   — Bayesian fit of plasma emission spectra including fully propagated uncertainties;
   Gaussian likelihood with per-line noise estimates; MCMC via emcee; directly analogous
   to LIBS application.

10. **Ciucci, A., Corsi, M., Palleschi, V., Salvetti, A. & Tognoni, E. (1999).** New
    Procedure for Quantitative Elemental Analysis by Laser-Induced Plasma Spectroscopy.
    *Applied Spectroscopy*, 53(8), 960–964. DOI: 10.1366/0003702991947612
    — Original CF-LIBS paper. Derives the closure equation (ΣCₛ = 1) and the self-absorption
    correction that any Bayesian forward model for LIBS must implement in F(x).

---

## 4. "What Correct Code MUST Do" Checklist

### Likelihood
- [ ] **Use signal-dependent noise**: `σₖ²(x) = λₖ(x) + σ_RON²` for each pixel k; do NOT
      use a single fixed σ across all pixels when LIBS signals span > 2 decades in intensity.
- [ ] **Include the log(σₖ²) normalisation term** in log-likelihood if computing WAIC, LOO,
      or Bayes factors (model comparison).
- [ ] **For Poisson-limited ICCD**: use proper Poisson log-likelihood (`y log λ − λ`) or the
      Poisson-Gaussian approximation; do not silently fall back to χ².

### Forward Model Integration
- [ ] **CF-LIBS closure inside the forward model**: `F(x)` must enforce ΣCₛ = 1 at every
      evaluation; the Bayesian code must not allow non-physical (negative or sum > 1)
      compositions to enter likelihood evaluation.
- [ ] **Saha ionisation balance**: the forward model must include all observed ionisation
      stages (I, II at minimum); using only neutral lines removes the nₑ–T diagnostic leverage
      and produces a degenerate posterior.
- [ ] **Partition functions U(T)**: must be evaluated at each T proposal, not fixed at the
      prior T; they enter the Boltzmann factor and affect both intensities and the OE Jacobian K.

### Composition Prior and Parameterization
- [ ] **Use ALR or ILR reparameterization** (unconstrained η ∈ ℝˢ⁻¹) for NUTS, OR
      `dist.Dirichlet` in NumPyro (which handles the simplex boundary automatically).
- [ ] **Do NOT** place independent Beta or Uniform[0,1] priors on each Cₛ independently;
      this violates the sum-to-one constraint and creates incorrect geometry.
- [ ] **Set concentration scale**: for Dirichlet, α = 1 (flat) is often too uninformative;
      α = (prior_mean * precision) where precision ~ 5–20 encodes compositional scale.

### Temperature and Density Priors
- [ ] **Use log-scale prior on T**: `log T ~ Normal(μ, σ)` with σ ≈ 0.3–0.5 in natural log;
      uniform on T creates a prior that is heavily weighted toward high T.
- [ ] **Use log-scale prior on nₑ**: electron density spans 10¹⁶–10¹⁸ cm⁻³; uniform in
      log-space is appropriate; constrain using Stark FWHM measurement as informative prior.

### NUTS / HMC Configuration (NumPyro)
- [ ] **Warm-up ≥ 500 steps** with dual-averaging adaptation (`target_accept_prob=0.80`).
- [ ] **Mass matrix initialisation**: use `init_to_value` with MAP estimate (from classical
      CF-LIBS solver or OE iterate) as starting point to speed up warm-up convergence.
- [ ] **Diagnose convergence**: compute R̂ and ESS for all parameters; flag R̂ > 1.05 or
      ESS < 50 as convergence failures; do NOT report posteriors from unconverged chains.
- [ ] **Report credible intervals**: use 68% or 95% HPD / equal-tailed intervals from the
      posterior samples, NOT just the MAP point estimate; this is the primary value of MCMC.

### Rodgers OE (when used as fast first-pass)
- [ ] **Jacobian K = ∂F/∂x** must be computed at current iterate, not fixed at prior xₐ;
      for CF-LIBS the Boltzmann/Saha K has strong T-dependence.
- [ ] **Include Sₐ regularisation**: pure LS (Sₐ → ∞) is ill-conditioned for CF-LIBS due
      to near-degenerate line ratios; always include a finite prior covariance.
- [ ] **Check averaging kernel A = Ŝ Kᵀ Sₑ⁻¹ K**: d_s = tr(A) < S (number of elements)
      warns that some composition components are unconstrained; do not report those as
      accurate retrievals.
- [ ] **Do NOT conflate posterior covariance Ŝ with measurement uncertainty**: Ŝ is the
      combined measurement + prior uncertainty; quote both the measurement contribution
      and the prior contribution separately for interpretability.
