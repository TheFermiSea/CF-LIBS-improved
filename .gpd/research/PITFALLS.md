# Physics and Computational Pitfalls

**Physics Domain:** GPU-accelerated CF-LIBS (Calibration-Free Laser-Induced Breakdown Spectroscopy)
**Researched:** 2026-03-23
**Confidence:** MEDIUM (web search unavailable; grounded in codebase inspection and established domain knowledge)

## Critical Pitfalls

Mistakes that invalidate results or waste months of computation.

### Pitfall 1: Humlicek W4 Gradient Instability in JAX Autodiff

**What goes wrong:** The Humlicek W4 rational approximation uses nested `jnp.where` to select among four regions of the complex plane. JAX evaluates ALL branches during reverse-mode AD (backpropagation), meaning gradients from region 4 (which contains `jnp.exp(u)` where u = t*t can be large) propagate NaN/Inf into the gradient even when that branch is not selected for the forward pass. This manifests as NaN gradients at high electron densities (log_ne > 17.5) where the Lorentzian component dominates and the Voigt parameter enters region boundaries.

**Why it happens:** JAX's AD traces all branches of `jnp.where` and combines them post-hoc. The `exp(u)` in region 4 overflows in float32 for |x| > ~8, producing Inf that contaminates gradients even when the forward value is correct (selected from a different region).

**Consequences:** MCMC sampling (NUTS) diverges silently. Joint optimizer (L-BFGS-B) returns garbage concentrations. The forward model may look correct while gradients are completely wrong.

**Warning signs:**
- NaN in gradient norms during optimization
- NUTS divergent transitions concentrated at high n_e
- Bayesian posteriors that exclude physically reasonable n_e values
- Loss landscape appears flat (zero gradients from NaN masking)

**Prevention:** The codebase already addresses this: `_faddeeva_humlicek_jax` is marked DEPRECATED in `profiles.py` line 612. The Weideman (1994) N=36 rational approximation (`_faddeeva_weideman_jax`) is branch-free by construction -- it uses a single rational function valid everywhere in the upper half-plane. **Use only the Weideman implementation for any JAX path.** Verify by checking that `_faddeeva_humlicek_jax` is never called in any active code path.

**Detection:** Run gradient checks: `jax.grad(forward_model)(params)` at (T=0.8 eV, n_e=1e18 cm^-3) and confirm no NaN. Unit test: compare `jax.grad` output against finite differences at extreme parameter corners.

**Which phase should address:** Voigt profile GPU port (Phase 1 or wherever profiles are ported to JAX).

**References:**
- Weideman, J.A.C. (1994) "Computation of the Complex Error Function", SIAM J. Numer. Anal. 31, 1497-1518.
- Humlicek, J. (1982) "Optimized computation of the Voigt and complex probability functions", JQSRT 27, 437-444.
- JAX issue tracker: known behavior that `jnp.where` evaluates both branches during AD.

---

### Pitfall 2: float32 Precision Loss in Voigt Profile Evaluation

**What goes wrong:** The Weideman N=36 approximation achieves ~15 digits of accuracy in float64. In float32 (mantissa = 23 bits = ~7.2 decimal digits), the polynomial evaluation in the Weideman coefficients suffers catastrophic cancellation: adjacent coefficients span 14 orders of magnitude (from ~5e-14 to ~2.7), and the partial sums oscillate in sign. The result is that the Voigt profile in float32 has relative errors of ~1e-3 to 1e-2 in the far wings, far exceeding the project's 1e-6 target.

**Why it happens:** Horner's method evaluating the 36-term polynomial accumulates roundoff. The smallest coefficients (~5e-14) are below float32 epsilon relative to the running sum, so they contribute nothing. Effectively, float32 reduces the Weideman approximation from N=36 to roughly N=20 effective terms.

**Consequences:** Wing intensities of strong lines are wrong by ~0.1-1%. For manifold generation where thousands of lines are summed, these wing errors accumulate. Boltzmann plot fits using wing-region intensities get biased temperatures.

**Warning signs:**
- Voigt profile values differ between scipy.special.wofz (float64) and JAX float32 path by more than 1e-6 relative
- Wing intensity tests fail for lines with large Lorentzian/Gaussian ratio (eta > 0.5)
- Systematic bias in round-trip validation temperature recovery

**Prevention:** **Use float64 on V100S GPUs.** V100S has full float64 throughput at 1/2 of float32 FLOPS (7.8 TFLOPS fp64 vs 16.4 TFLOPS fp32), which is exceptional among GPUs. The 2x slowdown is acceptable given the accuracy requirement. Set `jax.config.update('jax_enable_x64', True)` before any JAX import. The codebase already has the infrastructure for this in `jax_runtime.py` with `jax_default_real_dtype()`.

**Detection:** Automated regression test: evaluate Voigt at 100 points spanning wings to core, compare JAX result against scipy.special.wofz at float64. Assert max relative error < 1e-6. Test at multiple (sigma, gamma) ratios including gamma/sigma = 0.01, 1.0, 100.0.

**Which phase should address:** Voigt profile GPU port and validation phase.

**References:**
- Weideman (1994), Table 1: error analysis for different N values.
- NVIDIA V100S datasheet: 7.8 TFLOPS FP64 / 16.4 TFLOPS FP32.
- Goldberg, D. (1991) "What Every Computer Scientist Should Know About Floating-Point Arithmetic", ACM Computing Surveys 23(1), 5-48.

---

### Pitfall 3: Oscillator Strength Prefactor Unit Confusion (Latent Bug)

**What goes wrong:** The oscillator strength calculation in `self_absorption.py` (line 819-828) uses the prefactor 1.4992 with lambda in cm. The comment notes that "1.4992e-16 sometimes seen in literature is for lambda in Angstroms." If any downstream consumer passes wavelength in Angstroms instead of cm, or if the prefactor is copied to a new module with the wrong unit assumption, the optical depth is wrong by a factor of 1e16. The project context explicitly flags this: "oscillator_strength has latent 1e16 prefactor error (cancels in deltas)."

**Why it happens:** The relation A_ki = (8 pi^2 e^2 / m_e c lambda^2) * (g_i/g_k) * f_ik has the prefactor value depend on whether lambda is in cm (1.4992) or Angstroms (1.4992e-16). Papers use both conventions without always stating which.

**Consequences:** If the cancellation in deltas breaks (e.g., when computing absolute optical depth rather than ratios), self-absorption corrections are catastrophically wrong. Optical depths off by 1e16 mean every line appears either infinitely thick or infinitely thin.

**Warning signs:**
- Optical depth values that are absurdly large (>1e10) or absurdly small (<1e-20)
- Self-absorption corrections that have no effect OR that mask every single line
- Curve-of-growth slopes that disagree with expected linear/saturation transitions

**Prevention:** Add a dimensional analysis unit test: compute f_ik for a well-known line (e.g., Fe I 371.99 nm, A_ki = 1.62e8 s^-1, log(gf) = -0.43 from NIST) and assert the result matches the NIST log(gf) value to within 0.01 dex. Pin the unit convention in a module-level constant with an explicit docstring. Do NOT copy the prefactor to new code without the unit convention.

**Detection:** Cross-check computed gf values against NIST ASD for 10+ lines spanning UV to NIR. Any discrepancy > 0.1 dex indicates a unit error.

**Which phase should address:** Self-absorption correction phase; also relevant to any phase that computes absolute (not relative) optical depths.

**References:**
- NIST Atomic Spectra Database: https://physics.nist.gov/PhysRefData/ASD/lines_form.html
- Hilborn, R.C. (1982, corrected 2002) "Einstein coefficients, cross sections, f values, dipole moments, and all that", Am. J. Phys. 50, 982-986. arXiv:physics/0202029

---

### Pitfall 4: IPD Not Applied to Polynomial Partition Functions

**What goes wrong:** The Saha-Boltzmann solver computes ionization potential depression (IPD) via the Debye-Huckel model and passes `max_energy_ev=eff_ip_I` to `calculate_partition_function`. However, the polynomial partition functions (Irwin form: log U = sum a_n (log T)^n) in `partition.py` are pre-fitted to the FULL set of bound states up to the free-atom ionization limit. They do not accept or respect a `max_energy_ev` cutoff. At high n_e (>1e18 cm^-3), IPD can lower the effective ionization potential by 0.1-0.5 eV, removing high-lying Rydberg states from the partition sum. Ignoring this makes U(T) too large, biasing the Saha ratio.

**Why it happens:** Polynomial fits are compact and fast but encode the physics of ALL bound states. Truncating the sum at a lowered continuum requires either (a) explicit level-by-level summation with a cutoff, or (b) a correction term to the polynomial. Neither is implemented.

**Consequences:** At n_e = 1e18 cm^-3 and T = 1 eV, the partition function error for neutral iron is ~5-15%, propagating to a similar error in the ionization ratio and hence in derived concentrations. The error grows with n_e and with temperature (more Rydberg states populated).

**Warning signs:**
- Saha ratios that disagree with explicit level-by-level calculations at high n_e
- Partition function values that do not decrease when `max_energy_ev` is lowered
- Concentration results that shift systematically when n_e increases

**Prevention:** For the GPU-accelerated pipeline, implement a hybrid approach: use polynomial partition functions as a fast baseline, then apply an IPD correction factor. The correction is: U_corrected = U_poly - sum_{E_k > IP_eff} g_k * exp(-E_k / kT), summing over levels above the depressed continuum. Alternatively, pre-compute partition function tables on a (T, IP_eff) grid and interpolate.

**Detection:** Compare polynomial U(T) against explicit level-by-level summation (with and without IPD cutoff) for Fe I at T = 0.5, 1.0, 1.5 eV and n_e = 1e16, 1e17, 1e18. Report the fractional difference.

**Which phase should address:** Saha-Boltzmann solver GPU port phase. This is a physics correctness issue that must be resolved before manifold generation.

**References:**
- Irwin, A.W. (1981) "Polynomial partition function approximations of 344 atomic and molecular species", ApJS 45, 621.
- Stewart, J.C. & Pyatt, K.D. (1966) "Lowering of ionization potentials in plasmas", ApJ 144, 1203.
- Griem, H.R. (1964) "Plasma Spectroscopy", McGraw-Hill (Chapter 7: continuum lowering).

---

### Pitfall 5: LTE Validity Breakdown at Parameter Space Boundaries

**What goes wrong:** CF-LIBS assumes Local Thermodynamic Equilibrium (LTE): electron temperature governs all population distributions via Saha-Boltzmann statistics. The McWhirter criterion provides a necessary (but not sufficient) condition: n_e > 1.6e12 * T^0.5 * (Delta_E)^3 cm^-3, where Delta_E is the largest energy gap in eV. For typical LIBS plasmas (T ~ 0.5-2 eV, n_e ~ 1e16-1e18), LTE holds for most lines. But the manifold grid may extend to corners where LTE fails: low n_e with high T, or early-time gates before thermalization.

**Why it happens:** The manifold generator sweeps a parameter grid that may include (T, n_e) combinations where collisional rates cannot maintain equilibrium populations. The McWhirter criterion is checked in the CLI (line 340 of `main.py`) but the codebase has an `lte_validator.py` module that may not be enforced during manifold generation.

**Consequences:** Manifold points in non-LTE regions contain physically meaningless spectra. If these are used for inference (nearest-neighbor lookup), the retrieved parameters are wrong. Worse, the error is not obvious -- the spectra look reasonable, just with wrong relative line intensities.

**Warning signs:**
- McWhirter criterion violated: n_e < 1.6e12 * sqrt(T_K) * (Delta_E_eV)^3
- Boltzmann plot R^2 < 0.9 for well-known multiplets (indicates non-Boltzmann populations)
- Temperature derived from different elements disagree by > 20%
- Ionic-to-neutral line ratios inconsistent with Saha prediction

**Prevention:** Enforce LTE validity checks in the manifold generator. Mask or flag grid points that violate McWhirter. For the standard LIBS regime (n_e > 1e16, T = 0.5-2.0 eV), LTE is generally safe. Do not extend the grid below n_e = 5e15 cm^-3 without explicit non-LTE corrections.

**Detection:** Compute McWhirter criterion at each grid point using the maximum Delta_E for the elements in the composition. Log warnings for violations. In validation, check that round-trip temperature recovery degrades gracefully near LTE boundaries rather than failing silently.

**Which phase should address:** Manifold generation phase; validation phase should include LTE boundary tests.

**References:**
- McWhirter, R.W.P. (1965) in "Plasma Diagnostic Techniques", ed. Huddlestone & Leonard, Academic Press, Ch. 5.
- Cristoforetti, G. et al. (2010) "Local Thermodynamic Equilibrium in Laser-Induced Breakdown Spectroscopy: Beyond the McWhirter criterion", Spectrochimica Acta B 65, 86-95. DOI: 10.1016/j.sab.2009.11.005
- Aragón, C. & Aguilera, J.A. (2008) "Characterization of laser induced plasmas by optical emission spectroscopy: A review of experiments and methods", Spectrochimica Acta B 63, 893-916.

---

## Moderate Pitfalls

### Pitfall 6: Anderson Acceleration Divergence for Extreme T/n_e

**What goes wrong:** Anderson acceleration (AA) is planned for the Saha-Boltzmann fixed-point iteration. AA stores m previous iterates and computes an optimal linear combination to accelerate convergence. For extreme parameters (very high T with low n_e, or near ionization thresholds), the fixed-point map becomes near-singular: small changes in n_e produce large changes in ionization balance, and vice versa. AA with memory depth m > 3 can amplify these oscillations rather than damping them, leading to divergence or cycling.

**Prevention:**
1. **Start with m=2 or m=3.** Higher memory depths (m=5-10) are rarely beneficial for the 1D Saha-Boltzmann coupling and increase the risk of ill-conditioned least-squares problems in the AA update.
2. **Use damping (mixing parameter beta < 1).** beta = 0.5 is a safe starting point. Reduce to 0.3 if oscillations persist.
3. **Fall back to Picard iteration** (beta = 1, m = 0) if AA diverges after 5 iterations. Picard is slow but guaranteed to converge for contractive maps.
4. **Bound the iterate:** clamp n_e to [1e14, 1e20] and T to [0.1, 5.0] eV after each AA step to prevent unphysical excursions.
5. **Monitor the AA residual:** if ||F(x_k) - x_k|| increases for 3 consecutive steps, restart AA with m=1.

**Warning signs:**
- Residual oscillates or increases over iterations
- n_e or T jumps to extreme values between iterations
- AA least-squares condition number exceeds 1e10

**Detection:** Log the fixed-point residual and AA condition number at each iteration. Add a regression test at corner cases: (T=0.3, n_e=1e16), (T=3.0, n_e=5e15), (T=1.0, n_e=1e19).

**Which phase should address:** Anderson acceleration implementation phase.

**References:**
- Anderson, D.G. (1965) "Iterative procedures for nonlinear integral equations", J. ACM 12, 547-560.
- Walker, H.F. & Ni, P. (2011) "Anderson acceleration for fixed-point iterations", SIAM J. Numer. Anal. 49, 1715-1735. DOI: 10.1137/10078356X
- Toth, A. & Kelley, C.T. (2015) "Convergence analysis for Anderson acceleration", SIAM J. Numer. Anal. 53, 805-819.

---

### Pitfall 7: JIT Warmup and Async Dispatch Producing Misleading GPU Benchmarks

**What goes wrong:** JAX operations are dispatched asynchronously: `jnp.dot(a, b)` returns immediately while the GPU is still computing. Naive benchmarking with `time.time()` measures dispatch overhead (~microseconds), not actual compute time. Similarly, the first call to a `@jit`-compiled function includes tracing + XLA compilation (seconds), grossly misrepresenting steady-state performance. Reported "1000x speedups" often compare JIT-compiled GPU code (excluding warmup) against non-JIT CPU code (including Python overhead).

**Prevention:**
1. **Always call `block_until_ready()`** on the output before stopping the timer: `result = f(x); result.block_until_ready()`.
2. **Separate warmup from measurement.** Run the function once (or twice) to trigger JIT compilation, then benchmark subsequent calls.
3. **Report both JIT compilation time and per-call time.** Users need to know the one-time cost.
4. **Include host-device transfer time** when benchmarking end-to-end workflows. `jax.device_put()` and `jax.device_get()` can dominate for small arrays.
5. **Use `jax.profiler` or NVIDIA Nsight** for authoritative kernel timing, not wall-clock Python timers.
6. **Compare apples to apples:** benchmark NumPy (CPU, vectorized) against JAX (GPU, JIT) -- not Python loops against JAX.

**Warning signs:**
- GPU "speedup" exceeds theoretical FLOPS ratio (e.g., claiming 500x over NumPy when the V100S has ~50x peak FLOPS advantage over a single CPU core)
- Benchmark times that are suspiciously short (< 100 microseconds for non-trivial computation)
- Variance across runs that is > 50% of the mean (indicates JIT recompilation or GC interference)

**Detection:** The codebase has `cflibs/benchmark/unified.py` -- verify it calls `block_until_ready()`. Add a CI benchmark that asserts JIT warmup is excluded from reported times.

**Which phase should address:** Benchmarking and validation phase; also relevant during initial GPU port to avoid false confidence.

**References:**
- JAX documentation: "Asynchronous dispatch" section. https://jax.readthedocs.io/en/latest/async_dispatch.html
- JAX FAQ: "Benchmarking JAX code" section. https://jax.readthedocs.io/en/latest/faq.html

---

### Pitfall 8: Self-Absorption Correction Amplifying Noise on Weak Lines

**What goes wrong:** The self-absorption correction factor C = 1/f(tau) = tau / (1 - exp(-tau)) diverges as tau increases: C(1) = 1.58, C(2) = 2.31, C(3) = 3.16, C(5) = 5.03. For strong resonance lines with estimated tau > 3, the correction multiplies the measured intensity (and its noise) by 3-5x. If the optical depth estimate is itself uncertain (which it always is, since it depends on T, n_e, and composition -- the quantities being solved for), the correction can amplify systematic errors.

**Prevention:**
1. **Mask lines with tau > 3** (already the default `mask_threshold` in the codebase). Do not attempt to correct them; exclude from Boltzmann fits.
2. **Use doublet ratio methods** for self-absorption diagnostics before applying corrections. If the ratio of a known doublet deviates from its theoretical value, that quantifies self-absorption without needing T/n_e.
3. **Iterate conservatively:** apply self-absorption corrections only after a first pass that estimates T/n_e from optically thin lines. Do not use corrected intensities in the initial temperature determination.
4. **Propagate uncertainty:** the correction factor's uncertainty is delta_C/C ~ delta_tau / (1 - (1-exp(-tau))/tau), which exceeds 50% for tau > 2 with typical 30% uncertainty in tau.

**Warning signs:**
- Corrected intensities that are > 5x the measured values
- Temperature changing by > 10% when self-absorption correction is toggled on/off
- Corrected Boltzmann plot R^2 being WORSE than uncorrected

**Detection:** Compare T and concentrations with and without self-absorption correction. If they differ by > 15%, the correction is unreliable and should be replaced by line masking.

**Which phase should address:** Self-absorption module validation; inversion solver iteration design.

**References:**
- El Sherbini, A.M. et al. (2005) "Evaluation of self-absorption coefficients of aluminum emission lines in laser-induced breakdown spectroscopy measurements", Spectrochimica Acta B 60, 1573-1580.
- Bredice, F. et al. (2006) "Evaluation of self-absorption of manganese emission lines in LIBS", Spectrochimica Acta B 61, 1294-1303.

---

### Pitfall 9: NaN Propagation in JAX Silently Poisoning Batched Computations

**What goes wrong:** JAX operations propagate NaN silently: `jnp.sum([1.0, NaN, 3.0]) = NaN`. In a vmap-batched manifold generation, if ONE parameter combination produces a NaN (e.g., division by zero in the Saha equation at n_e = 0, or log(0) in the Boltzmann factor), it can poison an entire batch. Unlike NumPy which raises warnings, JAX in JIT mode produces no runtime warning.

**Prevention:**
1. **Enable JAX NaN checking during development:** `jax.config.update("jax_debug_nans", True)`. This raises an error on the first NaN but disables JIT (10-100x slower). Use only for debugging, not production.
2. **Guard inputs at the boundary:** clamp T > 0.1 eV, n_e > 1e10, partition functions > 1e-30. Use `jnp.maximum` not `jnp.clip` (clip has edge-case gradient issues in old JAX versions).
3. **Add NaN sentinels to manifold output:** after each batch, check `jnp.any(jnp.isnan(batch_result))` and log which parameter combinations failed. Replace NaN entries with interpolated values or mark as invalid.
4. **Use `jnp.where` with safe defaults:** `jnp.where(x > 0, jnp.log(x), -100.0)` instead of `jnp.log(jnp.maximum(x, 1e-30))` -- the latter still computes log(1e-30) = -69 which may cause downstream overflow when exponentiated.

**Warning signs:**
- Manifold HDF5 files containing NaN or Inf values
- Inference returning NaN concentrations for specific query spectra
- Mean/variance of manifold spectra showing unexpected NaN

**Detection:** Post-generation scan: `assert not np.any(np.isnan(manifold_data))`. Add this as a quality gate for manifold builds.

**Which phase should address:** Every phase that writes JAX code. Enforce as a coding standard from Phase 1.

**References:**
- JAX documentation: "Debugging NaNs" section. https://jax.readthedocs.io/en/latest/debugging/index.html

---

### Pitfall 10: Emissivity Units Mismatch Between Modules

**What goes wrong:** The project context flags: "emissivity docstring claims wrong units." The `spectrum_model.py` uses emissivity in the Kirchhoff relation (epsilon = kappa * B) where units must be consistent. If emissivity is in photon units (photons/s/cm^3/sr/nm) but the Planck function is in energy units (W/m^2/nm/sr), the self-absorption calculation produces wrong optical depths by a factor of h*nu. This factor varies across the spectrum (factor of ~2x between 200 nm and 800 nm), creating a wavelength-dependent systematic error.

**Prevention:**
1. **Audit every function's unit convention** and document in the docstring: specify whether intensity is spectral radiance (W/m^2/sr/nm), spectral irradiance (W/m^2/nm), photon spectral radiance (photons/s/m^2/sr/nm), or integrated line intensity (W/m^2/sr).
2. **Use SI consistently in the forward model:** W/m^2/sr/nm. Convert CGS atomic data (A_ki in s^-1, energies in eV, densities in cm^-3) at the module boundary, not inside inner loops.
3. **Add a dimensional analysis test:** compute the spectrum for a single strong line with known parameters and verify the absolute intensity matches an independent calculation (e.g., from NIST LIBS simulation).

**Warning signs:**
- Self-absorption corrections that have wrong wavelength dependence
- Absolute intensities that are off by factors of 1e4 or 1e7 (common CGS-SI conversion factors)
- Blackbody comparison failing by a wavelength-dependent factor

**Detection:** Compare `calculate_line_emissivity` output for Fe I 371.99 nm against a hand calculation using NIST values. The result should agree to < 1%.

**Which phase should address:** Forward model validation; should be resolved before self-absorption and manifold generation phases.

**References:**
- Griem, H.R. (1997) "Principles of Plasma Spectroscopy", Cambridge University Press, Ch. 4 (emissivity definitions).
- NIST LIBS simulation: https://physics.nist.gov/cgi-bin/ASD/lines1.pl (provides synthetic spectra in calibrated units for comparison).

---

## Minor Pitfalls

### Pitfall 11: Metal Backend float32-Only Restriction

**What goes wrong:** JAX Metal (Apple Silicon) does not support float64 or complex dtypes. The codebase correctly detects this and falls back to the real-arithmetic Weideman path (`_faddeeva_weideman_real_parts_jax`). However, developers testing on Mac laptops may not realize their accuracy is degraded compared to CUDA float64. Test suites that pass on Metal may fail the 1e-6 accuracy requirement on CUDA where float64 is expected.

**Prevention:** Test accuracy requirements are backend-conditional. Use `jax_backend_supports_x64()` to set tolerance: 1e-6 for float64 backends, 1e-3 for float32 backends. Document this in test fixtures.

**Which phase should address:** Testing infrastructure setup.

---

### Pitfall 12: Stark Broadening Estimation Fallback Inaccuracy

**What goes wrong:** When tabulated Stark parameters are unavailable, `stark.py` estimates w_ref using the hydrogenic approximation (effective quantum number from binding energy). For non-hydrogenic ions (most transition metals), this estimate can be off by 2-10x, producing wrong Voigt profiles for Stark-dominated lines.

**Prevention:** Log a warning when using estimated Stark parameters. Weight these lines lower in Boltzmann fits. Preferentially use lines with tabulated Stark parameters from Griem or the Stark-B database.

**Which phase should address:** Atomic data quality and line selection phases.

**References:**
- Griem, H.R. (1974) "Spectral Line Broadening by Plasmas", Academic Press.
- Sahal-Brechot, S. et al. (2015) "Stark-B database", https://stark-b.obspm.fr/

---

### Pitfall 13: Partition Function Polynomial Extrapolation Outside Fitted Range

**What goes wrong:** Irwin polynomial fits are calibrated for specific temperature ranges (typically 1000-16000 K). Extrapolating to T > 20000 K (T_eV > 1.7) can produce partition functions that decrease with temperature (unphysical) or diverge. The JAX implementation clamps T >= 1 K (`jnp.maximum(T_K, 1.0)`) but does not check the upper bound.

**Prevention:** Add upper temperature bounds from the original Irwin (1981) tables. For T above the fitted range, switch to explicit level summation or apply a high-T asymptotic correction. At minimum, assert dU/dT >= 0 (partition function is monotonically non-decreasing).

**Which phase should address:** Partition function module validation; manifold parameter grid design.

---

## Numerical Pitfalls

| Issue | Symptom | Cause | Fix |
|-------|---------|-------|-----|
| Catastrophic cancellation in Saha exponential | Ionization ratio jumps discontinuously | exp(-IP/kT) underflows to 0 for IP >> kT, then ratio * large_prefactor overflows | Compute log of Saha ratio first: log(S) = 1.5*log(T) + log(U_II/U_I) - IP/kT - log(n_e) + log(SAHA_CONST). Only exponentiate at the end. |
| Polynomial Horner evaluation roundoff | Partition function off by > 1% at T = 20000 K | Alternating-sign coefficients in Irwin fit, accumulated float32 roundoff | Use compensated (Kahan) summation, or evaluate in float64 even if the rest of the pipeline is float32. The partition function is evaluated once per species per temperature, so the cost is negligible. |
| Log-space overflow in Boltzmann factor | exp(-E_k / kT) = 0 for E_k >> kT | Upper levels with E_k > 15 * kT (~15 eV at T=1 eV) underflow in float64 | Work in log-space: log(n_k) = log(n_total) + log(g_k) - log(U) - E_k/kT. Only exponentiate to compute observables. |
| Grid aliasing in manifold wavelength grid | Missing narrow emission lines between grid points | Wavelength grid spacing > line FWHM / 3 | Ensure grid spacing < FWHM_min / (2 * sqrt(2 * ln(2))) for Nyquist sampling of the narrowest expected line. For R = 1000 at 200 nm, FWHM = 0.2 nm, need dx < 0.06 nm. |
| Denormalized floats in far wings | 1000x slowdown in wing computation on GPU | Profile values < 1e-38 (float32 denorm threshold) trigger microcode handling on NVIDIA GPUs | Flush wings to zero below a threshold: `profile = jnp.where(profile > 1e-30, profile, 0.0)`. Or use `--ftz=true` CUDA compiler flag (JAX does this by default on CUDA). |

## Convention and Notation Pitfalls

| Pitfall | Sources That Differ | Resolution |
|---------|-------------------|------------|
| Wavelength units: nm vs Angstrom vs cm | NIST uses Angstroms (vacuum/air); codebase uses nm; oscillator strength formula uses cm | All internal calculations in nm. Convert at I/O boundary. Mark every function parameter with units in docstring. |
| Air vs vacuum wavelengths | NIST reports air wavelengths above 200 nm, vacuum below | Use vacuum wavelengths internally. Apply Edlen (1966) correction when importing NIST data above 200 nm. |
| Electron density: cm^-3 vs m^-3 | SI uses m^-3; plasma spectroscopy literature universally uses cm^-3 | Use cm^-3 internally (matches all LIBS literature). Convert to SI only for Planck function and other SI-native calculations. Factor of 1e6 per conversion. |
| Temperature: eV vs K | Saha equation literature uses eV; partition function fits use K | Use eV as primary. Convert via EV_TO_K = 11604.52 K/eV. Document which convention each function expects. |
| Ionization stage numbering: 0-indexed vs 1-indexed | Spectroscopic notation: I = neutral, II = singly ionized (1-indexed). Some code uses 0 for neutral. | Codebase uses 1-indexed (spectroscopic convention). Verify at every interface. A +/- 1 error in ionization stage causes wrong partition function lookup. |
| Gaussian sigma vs FWHM | Some papers define Voigt in terms of FWHM_G, others use sigma | Codebase uses sigma (standard deviation) internally. FWHM = 2.355 * sigma. Document at every function boundary. |
| Lorentzian gamma: FWHM vs HWHM | Literature splits roughly 50/50 | Codebase uses HWHM (Half-Width at Half-Maximum). FWHM_L = 2 * gamma. The factor of 2 error is extremely common. |
| Saha constant value | Depends on unit system: SI, CGS, and whether n_e is in cm^-3 or m^-3 | Use `SAHA_CONST_CM3` from `constants.py`. Verify numerically: SAHA_CONST_CM3 = 2 * (2*pi*m_e*k_B)^(3/2) / h^3 = 4.829e15 cm^-3 eV^(-3/2). |

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Voigt profile GPU port | float32 precision loss (Pitfall 2); Humlicek gradient instability if old code is accidentally used (Pitfall 1) | Use float64 on V100S. Delete or clearly fence the deprecated Humlicek implementation. |
| Saha-Boltzmann GPU port | IPD not applied to polynomial partition functions (Pitfall 4); Saha exponential overflow (Numerical table) | Implement IPD correction for partition functions. Work in log-space for Saha ratios. |
| Anderson acceleration | Divergence at extreme parameters (Pitfall 6) | Start with m=2, beta=0.5. Implement Picard fallback. Clamp iterates to physical bounds. |
| Manifold generation | LTE validity at grid boundaries (Pitfall 5); NaN propagation in batched computation (Pitfall 9) | Enforce McWhirter checks. Add post-generation NaN scan. |
| Benchmarking | JIT warmup and async dispatch (Pitfall 7) | Use block_until_ready(). Separate warmup. Report both JIT time and steady-state time. |
| Self-absorption | Noise amplification on strong lines (Pitfall 8); unit mismatch in emissivity (Pitfall 10) | Mask tau > 3. Audit emissivity units before implementing correction. |
| Inversion solver | Oscillator strength unit confusion if absolute tau needed (Pitfall 3) | Validate gf values against NIST for 10+ lines before using absolute optical depths. |
| Forward model validation | Emissivity units (Pitfall 10); convention clashes (Convention table) | Dimensional analysis test against hand calculation. Pin all conventions in module-level docstrings. |

## Sources

- Weideman, J.A.C. (1994) SIAM J. Numer. Anal. 31, 1497-1518. [Faddeeva function computation]
- Humlicek, J. (1982) JQSRT 27, 437-444. [W4 Voigt approximation]
- Goldberg, D. (1991) ACM Computing Surveys 23(1), 5-48. [Floating-point arithmetic]
- Walker, H.F. & Ni, P. (2011) SIAM J. Numer. Anal. 49, 1715-1735. DOI: 10.1137/10078356X [Anderson acceleration]
- Cristoforetti, G. et al. (2010) Spectrochimica Acta B 65, 86-95. DOI: 10.1016/j.sab.2009.11.005 [LTE validity beyond McWhirter]
- Irwin, A.W. (1981) ApJS 45, 621. [Polynomial partition functions]
- Stewart, J.C. & Pyatt, K.D. (1966) ApJ 144, 1203. [Continuum lowering]
- Griem, H.R. (1974) "Spectral Line Broadening by Plasmas", Academic Press. [Stark broadening]
- Griem, H.R. (1997) "Principles of Plasma Spectroscopy", Cambridge University Press. [Emissivity, plasma diagnostics]
- Hilborn, R.C. (2002) Am. J. Phys. 50, 982. arXiv:physics/0202029 [Einstein coefficients and oscillator strengths]
- El Sherbini, A.M. et al. (2005) Spectrochimica Acta B 60, 1573-1580. [Self-absorption evaluation]
- Aragón, C. & Aguilera, J.A. (2008) Spectrochimica Acta B 63, 893-916. [CF-LIBS review]
- JAX documentation: Asynchronous dispatch, Debugging NaNs sections. https://jax.readthedocs.io/en/latest/
- NVIDIA V100S datasheet. [FP64/FP32 throughput ratios]
- NIST Atomic Spectra Database. https://physics.nist.gov/PhysRefData/ASD/

**Note:** Web search was unavailable during this research session. All findings are grounded in codebase inspection and established domain knowledge (textbook-level plasma physics, well-documented JAX behavior, published numerical methods). Confidence is MEDIUM rather than HIGH due to inability to verify recent library changes or check for recent errata. Recommend re-running literature verification searches when web access is restored.
