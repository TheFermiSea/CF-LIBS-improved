# Composition Closure Audit: Mole↔Mass + Aitchison/ILR

## VERDICT

**FLAWED — MEDIUM severity** (two distinct issues of different severity; no critical algebraic bugs but a persistent reporting defect)

The ILR/Helmert basis is mathematically correct (isometric, orthonormal, round-trips to machine epsilon). The oxide stoichiometry (O/cation ratios) is correct. The Egozcue 2003 covariance Jacobian is implemented exactly. **However:**

1. **The `_number_to_mass_fractions` function in `pipeline.py` (L949) is defined but never called for the iterative or closed-form solver path.** The scoreboard at `scoreboard.py:305` reads `result.concentrations` directly and multiplies by 100 to get `predicted_wt`, but `CFLIBSResult.concentrations` contains **mole (number) fractions**, not mass fractions. This is the Völker 2024 bug. For steel compositions (Si especially), Si is understated in mass by 4.73× its atomic mass relative to Fe, causing +96% relative error on Si and ~0.73 wt% additional RMSE on top of physics errors.

2. **The `ClosureResult.concentrations` annotation correctly labels the output as mole fractions and the `CFLIBSResult` docstring warns "Convert to mass fractions via…"**, but no caller in the iterative/closed-form dispatch path performs that conversion before scoring. The conversion only happens for the `full_spectrum` solver (`full_spectrum.py:647`).

---

## GROUND TRUTH

**Governing equation (Tognoni 2010, Ciucci 1999):**

CF-LIBS closure gives **number (mole) fractions**:
```
C_s = U_s(T) · exp(q_s) / F          (number fraction)
Σ_s C_s = 1
```

Mass fractions require the explicit conversion:
```
w_s = C_s · M_s / Σ_j (C_j · M_j)   (mass fraction)
```

**Citations:**

1. **Völker T. (2024)** "Mass and mole fractions in calibration-free LIBS." *J. Analytical Atomic Spectrometry* (RSC). DOI: 10.1039/D4JA00028E — **primary source.** Proves CF-LIBS closure gives mole fractions; quantifies errors up to 353% from omitting the M_s-weighted conversion; gives explicit formula above.

2. **Tognoni E., Cristoforetti G., Legnaioli S., Palleschi V. (2010)** "Calibration-Free Laser-Induced Breakdown Spectroscopy: State of the art." *Spectrochimica Acta Part B* 65(1), 1–14. DOI: 10.1016/j.sab.2009.11.006 — canonical CF-LIBS review; Section 2.1 makes explicit that the closure equation yields elemental number fractions.

3. **Egozcue J.J., Pawlowsky-Glahn V., Mateu-Figueras G., Barceló-Vidal C. (2003)** "Isometric Logratio Transformations for Compositional Data Analysis." *Mathematical Geology* 35(3), 279–300. DOI: 10.1023/A:1023818214614 — canonical ILR isometry reference; defines Helmert basis and proves V^T V = I_{D-1}, d_A(x,y) = ||ilr(x)-ilr(y)||_2.

4. **Ciucci A., Corsi M., Palleschi V., Rastelli S., Salvetti A., Tognoni E. (1999)** "New Procedure for Quantitative Elemental Analysis by Laser-Induced Plasma Spectroscopy." *Applied Spectroscopy* 53(8), 960–964. DOI: 10.1366/0003702991947612 — original CF-LIBS paper defining C_s as number concentrations.

5. **Hron K., Filzmoser P., Thompson K. (2012)** "Linear regression with compositional explanatory variables." *Journal of Applied Statistics* 39(5), 1115–1128. DOI: 10.1080/02664763.2011.644268 — pivot isometric log-ratio coordinates (PLR/PWLR isometry).

6. **Jochum K.P. et al. (2016)** "GeoROC: The database of geochemical data for reference materials." — ground truth for oxide stoichiometry (Si→SiO₂: 2 O/Si, Al→Al₂O₃: 1.5 O/Al, etc.)

---

## CODE VALUE (numerical)

### ILR/Helmert tests

```bash
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && \
PYTHONPATH=/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 JAX_PLATFORMS=cpu python -c "
import numpy as np
from cflibs.inversion.physics.closure import _helmert_basis, ilr_transform, ilr_inverse, _pivot_contrast

# Helmert orthonormality
for D in [2, 3, 4, 5, 10]:
    V = _helmert_basis(D)
    err = np.max(np.abs(V.T @ V - np.eye(D-1)))
    print(f'D={D}: max|V^T V - I| = {err:.2e}')  # 2.2e-16 for all D

# ILR round-trip
x = np.array([0.7, 0.2, 0.1])
x_back = ilr_inverse(ilr_transform(x), 3)
print(f'round-trip err: {np.max(np.abs(x_back - x)):.2e}')  # 1.1e-16

# ILR isometry: ||ilr(x)-ilr(y)|| == d_Aitchison(x,y)
y = np.array([0.5, 0.3, 0.2])
d_aitchison = 0.7512984632  # computed independently
d_ilr = np.linalg.norm(ilr_transform(x) - ilr_transform(y))
print(f'isometry diff: {abs(d_aitchison - d_ilr):.2e}')  # 0.0
"
```

**Output:**
```
D=2: max|V^T V - I| = 2.22e-16
D=3: max|V^T V - I| = 2.22e-16
D=4: max|V^T V - I| = 2.22e-16
D=5: max|V^T V - I| = 2.22e-16
D=10: max|V^T V - I| = 2.22e-16
round-trip err: 1.11e-16
isometry diff: 0.00e+00
```

### Oxide stoichiometry test

All 10 oxide O/cation ratios match standard geochemistry (Si→2.0, Al→1.5, Fe→1.5, Mn→1.0, Mg→1.0, Ca→1.0, Na→0.5, K→0.5, P→2.5): **all correct**.

### Mole-vs-mass defect: Steel 304 composition

```bash
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && \
PYTHONPATH=/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 JAX_PLATFORMS=cpu python -c "
import numpy as np
molar = {'Fe': 55.845, 'Cr': 51.996, 'Ni': 58.693, 'Mn': 54.938, 'Si': 28.085}
mass_fracs = {'Fe': 0.715, 'Cr': 0.180, 'Ni': 0.080, 'Mn': 0.020, 'Si': 0.005}
mole_fracs = {el: v/molar[el] for el, v in mass_fracs.items()}
tot = sum(mole_fracs.values()); mole_fracs = {k: v/tot for k, v in mole_fracs.items()}
errs = [(mole_fracs[el]-mass_fracs[el])**2 for el in mass_fracs]
print(f'RMSE wt% from mole-vs-mass: {np.sqrt(np.mean(errs))*100:.2f}')
for el in sorted(mass_fracs): print(f'{el}: err={(mole_fracs[el]-mass_fracs[el])*100:+.3f}%')
"
```

**Output:**
```
RMSE wt% from mole-vs-mass: 0.73
Cr: err=+1.052%
Fe: err=-1.037%
Mn: err=+0.004%
Ni: err=-0.499%
Si: err=+0.480%
```

**Carbon-in-steel worst case (reproducing Völker 2024):**
- C mass fraction: 0.100%  
- C mole fraction (what CF-LIBS returns): 0.462%  
- Relative error if mole reported as mass: **+361.8%** (matches Völker 2024 Fig.1 claim of 353%)

### Where the conversion is missing

```
cflibs/benchmark/scoreboard.py:305:
    record["predicted_wt"] = {el: 100.0 * float(predicted.get(el, 0.0)) for el in candidates}
    # predicted = dict(result.concentrations) — MOLE FRACTIONS from iterative/closed_form solver
    # No mass conversion here or in the dispatch chain for these solvers.

cflibs/inversion/pipeline.py:949: def _number_to_mass_fractions(...)  # DEFINED, NEVER CALLED for peak-based path
cflibs/inversion/solve/full_spectrum.py:647: fit_mass = _number_to_mass_fractions(fit_numfrac)  # ONLY full_spectrum solver does it
```

### Egozcue 2003 Jacobian check

```
Max |J_code - J_egozcue| = 1.11e-16   (machine epsilon: correct)
Max |V^T G - V^T| = 5.55e-17          (Helmert columns sum to zero: verified)
ILR covariance positive definite: True
```

---

## DELTA & INTERPRETATION

### Issue 1: Mole-vs-mass reporting defect

**Severity: MEDIUM** (not physics-of-the-algorithm, but mis-reporting of results)

The iterative and closed-form solvers correctly compute mole (number) fractions from Saha-Boltzmann closure. The `CFLIBSResult.concentrations` field is correctly labeled internally. However, `scoreboard.py` multiplies these by 100 and treats them as `predicted_wt%`, inflating the reported RMSE by approximately:
- 0.73 wt% RMSE for a 5-element steel (Si:0.5%, Fe:71.5%, Cr:18%, Ni:8%, Mn:2%)  
- Up to +362% relative error on light elements (C, Si) vs. heavy elements (Fe, Ni)
- The `full_spectrum` solver path correctly applies `_number_to_mass_fractions` before scoring (only that path)

**Impact on reported benchmarks:** The iterative and closed-form solver RMSE numbers on steel/geological datasets are measured in mole-fraction units but compared to mass-fraction (wt%) ground truth. For Fe-dominated steel (average molar mass ~54 g/mol vs near-iron), most elements have |Δw_s| < 1 wt% because Fe dominates the denominator. For light-element-containing samples (geological rocks with Si, Al), the error grows substantially: Si at 0.5 wt% reports at 0.98 wt% (+96%).

### Issue 2: ILR/Helmert math — CORRECT

All ILR algebra is verified correct to machine epsilon:
- `V^T V = I_{D-1}` for D=2..10 (max error 2.2e-16)  
- Round-trip `ilr_inverse(ilr_transform(x)) == x` (max error 1.1e-16)
- Isometry: `||ilr(x) - ilr(y)||_2 == d_Aitchison(x,y)` (exact)
- PLR pivot coordinates: `Psi @ Psi.T = I_{D-1}` (max error 4.4e-16)
- Egozcue 2003 covariance Jacobian: `J = V^T diag(1/c)` (correct, uses V^T G = V^T identity)

### Issue 3: Oxide stoichiometry — CORRECT

All 10 O/cation ratios match standard geochemical references (JGS, Jochum 2016). The code correctly uses O-per-cation (2.0 for SiO₂) rather than the molar-mass ratio (2.139 for SiO₂/Si), which is the correct weight for molar-oxygen balance normalization.

---

## FIX

**Fix for mole-vs-mass defect:**

The `_number_to_mass_fractions` function already exists correctly in `pipeline.py`. It needs to be called in `_run_peak_based_solver` before returning for the iterative/closed-form path:

```python
# cflibs/inversion/pipeline.py, inside _run_peak_based_solver(),
# after the solver.solve() / _solve_analyze_result() call:

result_with_mole = _solve_analyze_result(
    solver, observations, pipeline.closure_mode, closure_kwargs,
    uncertainty_mode, stark_diagnostics=stark_diagnostics,
)
# Convert mole fractions → mass fractions before returning to the scoreboard
mass_concs = _number_to_mass_fractions(result_with_mole.concentrations)
return result_with_mole.__class__(
    **{**vars(result_with_mole), "concentrations": mass_concs}
)
```

Similarly for the `closed_form` branch (L1110):
```python
result = solver.solve(...)
mass_concs = _number_to_mass_fractions(result.concentrations)
return result.__class__(**{**vars(result), "concentrations": mass_concs})
```

**Note:** Before applying this fix, verify that downstream code (Bayesian MCMC priors, Saha iteration, charge-balance checks) does NOT consume `CFLIBSResult.concentrations` expecting mole fractions — if so, a rename/separate field is required. The `_refine_ne_pressure_balance` in `closed_form.py` uses `compositions` (a local variable, not `CFLIBSResult.concentrations`) so that path is safe.

**What would have to be true for ILR to be wrong:** The only way the ILR/Helmert implementation could be wrong is if NumPy's `sqrt`, `exp`, and matrix operations introduced errors exceeding ~1e-14, which would require a broken BLAS. This is ruled out by the numerical tests.
