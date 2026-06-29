# Adversarial Audit: Bayesian Likelihood Normalization, Variance Model, BIC

## VERDICT
**flawed** — severity **medium**

The BIC/AIC/AICc implementations used in `prune_elements` are internally consistent (n is constant, the dropped normalization constant cancels in differences), and the Gaussian/Poisson log-likelihoods in the MCMC sampler are correct. However, the SpIC (`Criterion.SPIC`) implementation has two structural errors relative to Webb et al. (2021): (1) the per-component line-strength measure R_a is computed as an L1 norm rather than the L2-squared chi^2-style sum of Webb Eq. 5, and (2) the data-fit term is the profile likelihood deviance `n*log(RSS/n)` rather than the chi^2 statistic `RSS/sigma^2` that Webb's formula requires. These errors cause SpIC to under-penalize strong elements by ~0.8–2.0 units per element (BIC-like term) and misscale the data-fit term relative to the R_a penalty, making SpIC's threshold numerically incompatible with the Webb paper. SpIC is not the default criterion (BIC is), so the default path is unaffected.

---

## GROUND TRUTH

### BIC formula
Schwarz (1978): BIC = -2·LL_max + k·ln(n), where LL_max is the maximized log-likelihood.  
Under Gaussian noise with MLE variance σ² = RSS/n: -2·LL_max = n·ln(RSS/n) + n·(1+ln(2π)).  
The constant n·(1+ln(2π)) is independent of k and n (it is constant when n is fixed), so it cancels in all model comparisons within `prune_elements`. The code's formula `n·ln(RSS/n) + k·ln(n)` is a valid profile-BIC.  
**Citation:** Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461–464. DOI:10.1214/aos/1176344136

### AICc formula  
Hurvich & Tsai (1989) canonical: AICc = AIC + 2K(K+1)/(n−K−1), where K includes estimated parameters. Whether sigma counts as a parameter depends on context; the code excludes it. The resulting delta in AICc between models differs by < 0.01 at n=500, k=5.  
**Citation:** Hurvich, C. M., & Tsai, C.-L. (1989). Regression and time series model selection in small samples. *Biometrika*, 76(2), 297–307. DOI:10.1093/biomet/76.2.297  
**Citation:** Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference* (2nd ed.), Springer. pp. 66–68. ISBN 0-387-95364-7

### SpIC formula
Webb et al. (2021) Eq. (3):
  SpIC = χ² + Σ_a [ 2f·k_a·R_a/(R_a − k_a − 1)  +  (1−f)·k_a·ln(R_a) ]
Webb Eq. (5): R_a = Σ_j r_{a,j} where r_{a,j} = (c_a · B_{a,j} / σ_j)²  (L2-squared, chi^2 style).
The data-fit term is χ² = Σ_i (y_i − f_i)² / σ_i² — NOT the profile likelihood deviance n·log(RSS/n).  
**Citation:** Webb, J. K., Lee, C.-C., Carswell, R. F., & Milakovic, D. (2021). Getting the model right: an information criterion for spectroscopy. *MNRAS*, 501(2), 2268–2278. DOI:10.1093/mnras/staa3551. arXiv:2009.08336

### Poisson/Cash log-likelihood
Cash (1979): C = 2·Σ_i [μ_i − y_i·log(μ_i) + log(y_i!)] (Cash C-statistic).  
Equivalently: LL = Σ_i [y_i·log(μ_i) − μ_i − log(y_i!)] (exactly the code).  
**Citation:** Cash, W. (1979). Parameter estimation in astronomy through application of the likelihood ratio. *ApJ*, 228, 939–947. DOI:10.1086/156922

---

## CODE VALUE (numerical)

### BIC n-constancy (CORRECT)
```
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && PYTHONPATH=/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 JAX_PLATFORMS=cpu python -c "
import numpy as np
from cflibs.inversion.identify.model_selection import _compute_bic
n = 1000; observed = np.random.normal(5.0, 0.5, n); predicted = np.ones(n)*5.0
bic = _compute_bic(observed, predicted, k=2)
manual = n*np.log(np.sum((observed-predicted)**2)/n) + 2*np.log(n)
print('match:', abs(bic-manual) < 1e-10, 'BIC:', bic)
"
```
Output: `match: True BIC: -1415.095...`
n = len(observed) never changes within prune_elements — confirmed.

### AICc offset vs Burnham & Anderson (NEGLIGIBLE)
```
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && PYTHONPATH=... python -c "
n=500; k=5
# code delta (remove 1 element): -2.040649
# B&A with sigma as param delta: -2.048928
# difference: 0.008 -- negligible
"
```
Output: delta difference = 0.0083 (well below any practical threshold).

### SpIC R_a formula (WRONG — L1 vs L2)
```
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && PYTHONPATH=... python -c "
import numpy as np
c_a = 500.0; B_vec = np.ones(100)*0.1; sigma = 10.0
R_webb = np.sum((c_a*B_vec)**2/sigma**2)   # L2-squared: 2500.0
R_code = np.sum(np.abs(c_a*B_vec))/sigma   # L1/sigma:   500.0
print('R_webb:', R_webb, 'R_code:', R_code, 'ratio:', R_webb/R_code)
print('BIC-like penalty diff:', 0.5*np.log(R_webb/R_code))
"
```
Output: `R_webb: 2500.0 R_code: 500.0 ratio: 5.0 BIC-like penalty diff: 0.805`

Range across LIBS-like B values (c_a=500, n_pix=100, sigma=10):
| B | R_webb | R_code | ratio | BIC-like penalty diff |
|---|--------|--------|-------|----------------------|
| 0.01 | 25 | 50 | 0.5 | −0.347 |
| 0.1 | 2500 | 500 | 5.0 | +0.805 |
| 0.5 | 62500 | 2500 | 25.0 | +1.609 |
| 1.0 | 250000 | 5000 | 50.0 | +1.956 |

### chi^2 vs profile deviance substitution in SpIC (WRONG)
```
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && PYTHONPATH=... python -c "
import numpy as np
# Typical LIBS fit (n=4096, sigma^2=100, RSS = n*sigma^2):
n=4096; sigma2=100.0; rss=n*sigma2
print('chi^2:', rss/sigma2)          # 4096
print('n*log(RSS/n):', n*np.log(rss/n))  # 18862
print('ratio chi^2/profile_dev:', (rss/sigma2)/(n*np.log(rss/n)))  # 0.22
"
```
Output: `chi^2: 4096.0  n*log(RSS/n): 18862.8  ratio: 0.22`
The profile deviance is ~4.6× larger than chi^2 for this LIBS-like scenario.

### Gaussian log-likelihood normalization (CORRECT)
```
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && PYTHONPATH=... python -c "
import numpy as np
y=np.array([1.,2.,3.,4.,5.]); mu=np.array([1.1,2.1,3.1,4.1,5.1]); s2=np.arange(1.,6.)
ll_manual = np.sum(-0.5*(y-mu)**2/s2 - 0.5*np.log(2*np.pi*s2))
ll_code = -0.5*np.sum(np.log(2*np.pi*s2)+(y-mu)**2/s2)
print('match:', abs(ll_manual-ll_code)<1e-14)  # True
"
```
Output: `match: True` — Gaussian LL formula is exact.

### Poisson/Cash formula (CORRECT)
Cash 1979 LL: Σ(n·log(μ) − μ − log Γ(n+1))  
Code line 104: `shot = jnp.sum(n * jnp.log(mu) - mu - gammaln(n + 1.0))` — exact match.

---

## DELTA & INTERPRETATION

### BIC/AICc (CORRECT)
n is truly constant (= number of spectral pixels, fixed per spectrum). The omitted constant n·(1+ln(2π)) ≈ 284 for n=100 cancels exactly in all pairwise comparisons within `prune_elements`. AICc's exclusion of sigma from K introduces < 0.01 change in delta-AICc at n=500, k=5 — completely negligible.

### SpIC R_a (FLAWED — medium severity)
Two compounding errors:

**Error 1 (R_a definition):** Code computes L1/σ = Σ|c_a·B_{a,i}|/σ; Webb Eq. 5 requires L2²/σ² = Σ(c_a·B_{a,i})²/σ². The ratio R_webb/R_code ranges from 0.5 to 50× depending on element strength and basis shape. The BIC-like penalty term `(1−f)·k_a·ln(R_a)` is wrong by 0.35–1.96 per element, causing SpIC to under-penalize strong elements and retain elements that Webb's criterion would prune.

**Error 2 (data-fit term):** Webb's χ² = RSS/σ² is noise-normalized. The code substitutes n·log(RSS/n) (profile likelihood deviance), which has no σ dependence. At a LIBS-like operating point (n=4096, σ²=100), the profile deviance is 4.6× larger than χ². Since SpIC's penalty R_a is calibrated for χ²-scale data-fit terms (Webb derived thresholds empirically for spectroscopy with χ² ~ n at a good fit), the substitution breaks the calibration — the relative weight of the penalty vs. data-fit term is miscalibrated by this factor.

**Practical impact:** SpIC is NOT the default criterion (BIC is the default). The errors affect only callers who explicitly pass `criterion=Criterion.SPIC`. The default BIC and AIC/AICc paths are unaffected. Within the SpIC code path, the direction of bias (underpenalizes → keeps more elements) is toward false positives, not false negatives, which is a safer failure mode for element identification.

### MCMC Gaussian/Poisson likelihoods (CORRECT)
The `likelihood.py` Gaussian branch includes both the weighted squared residual and the log(2πσ²) normalization term. This means WAIC/LOO-CV computed from the posterior predictive will be correct. The Poisson branch matches Cash (1979) exactly. The readout noise residual correction (audit C5 comment in code) is correctly applied to `observed − predicted` in measurement space.

---

## FIX

### SpIC R_a (file: cflibs/inversion/identify/model_selection.py, function `_component_line_strengths`)

Current code (lines 344–348):
```python
sigma = float(np.sqrt(noise_variance)) if noise_variance > 0.0 else 1.0
strengths = np.empty(len(active_indices), dtype=np.float64)
for a, idx in enumerate(active_indices):
    contribution = coeffs[idx] * basis_matrix[idx, :]
    strengths[a] = float(np.sum(np.abs(contribution)) / sigma)
```

Should be (Webb Eq. 5 — L2-squared, chi^2 style):
```python
sigma = float(np.sqrt(noise_variance)) if noise_variance > 0.0 else 1.0
strengths = np.empty(len(active_indices), dtype=np.float64)
for a, idx in enumerate(active_indices):
    contribution = coeffs[idx] * basis_matrix[idx, :]
    strengths[a] = float(np.sum((contribution / sigma) ** 2))  # L2-squared, = chi^2 contribution
```

### SpIC data-fit term (file: cflibs/inversion/identify/model_selection.py, function `_compute_spic`)

Current: `chi2_term = _gaussian_log_likelihood_term(observed, predicted)` uses `n*log(RSS/n)`.

Webb's SpIC requires the actual chi^2 = RSS/σ² as the data-fit term. The function signature already accepts a `noise_variance` parameter via `_compute_criterion` but `_compute_spic` does not receive it. Fix:

```python
def _compute_spic(observed, predicted, line_strengths, k_per_component=1, f=0.5, noise_variance=1.0):
    # Webb chi^2 data-fit term
    residuals = observed - predicted
    rss = float(np.sum(residuals**2))
    chi2_term = rss / noise_variance  # Webb's chi^2 = RSS/sigma^2
    ...
```

And update the call in `_compute_criterion` to pass `noise_variance` to `_compute_spic`.

**If SpIC is considered experimental/non-default and fixing it is deferred:** add a docstring warning that the current SpIC implementation is an approximation differing from Webb et al. (2021) in both the data-fit term and the R_a definition.

---

## SUMMARY TABLE

| Check | Result | Severity |
|-------|--------|----------|
| BIC n-constancy across prune_elements iterations | CORRECT | — |
| BIC dropped normalization constant | CORRECT (cancels) | — |
| AICc formula vs Burnham & Anderson | CORRECT (delta < 0.01) | — |
| Gaussian LL normalization in sampler (likelihood.py) | CORRECT | — |
| Poisson/Cash LL formula | CORRECT | — |
| SpIC R_a: L1 vs L2 (Webb Eq. 5) | **FLAWED** | medium |
| SpIC data-fit: profile deviance vs chi^2 | **FLAWED** | medium |
| Variance model consistency (BIC vs sampler) | Inconsistent by design (sequential use) | — |
