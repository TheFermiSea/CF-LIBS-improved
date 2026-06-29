This is a synthesis task. The three audits are unanimous and well-cited. Let me write the verdict directly.

# CF-LIBS Boltzmann-Plot Ordinate Convention: Literature-Grounded Verdict

## 1. Verdict — which ordinate is the standard, and is either repo *wrong*?

**Our numerical pipeline — `y = ln(I·λ / (g_k·A_ki))` vs `x = E_k` — is the literature-standard CF-LIBS Boltzmann/Saha-Boltzmann ordinate. CORRECT.**

This is not a judgment call; the standard form is printed verbatim across independent foundational and recent sources, all in agreement that the λ belongs in the numerator *whenever `I` is a measured energy/radiance quantity* (which is what every real spectrometer integrates):

- **Aragón & Aguilera, *Spectrochim. Acta B* 63 (2008) 893–916** — the canonical Saha-Boltzmann-plot review. Line intensity in **energy units** with the per-photon `hc/λ` factor; ordinate `ln(I λ / (g_k A_ki))` vs `E_k`, slope `−1/k_BT`.
- **Alrebdi et al., *Molecules* (2022), PMC9229630** — prints the working ordinate **verbatim**: `ln(I_ij·λ_ij / (h c A_ij g_i)) = −E_k/(k_B T_e) + ln(N/U(T)…)`. λ explicitly in the numerator; `hc` explicit.
- **Khelladi et al., *EPJ Appl. Phys.* 101 (2023) ap230072** — integrated intensity `J_tot = n_i·A_ij·(hc/λ)/(4π)`, Boltzmann plot `ln(J_tot·λ/(A_ij·g_i)) = ln(hc/(4π·n·Z)) − E_i/(kT)`. The λ multiplies `J` **precisely to cancel the `1/λ` photon-energy factor** — this is exactly our pipeline's form, shown explicitly.
- **NIST Atomic Spectroscopy compendium** — line radiance `I = (1/4π)(hc/λ₀) A_ki N_k ℓ`; the `hc/λ` is per-photon energy × spontaneous-emission rate. `ln(I·λ/(g_k A_ki))` removes the `1/λ` and yields the affine-in-`E_k` line.
- **"Importance of physical units in the Boltzmann plot method," *J. Anal. At. Spectrom.* (2022) DOI 10.1039/D2JA00241H** — decisive on units: intensity must be in **energy-flux (W m⁻²)**; the per-photon `hc/λ_ki` factor is present in Eq. (1).
- Original **Ciucci et al., *Appl. Spectrosc.* 53 (1999) 960** and the **Tognoni et al. *Spectrochim. Acta B* 65 (2010) 1** review are consistent with all of the above.

The `ASSERT_CONVENTION` markers in `cflibs/inversion/physics/boltzmann.py` (lines 56, 123, 689, 1400) and `boltzmann_jax.py:11` enforce the correct form. **No change to the shipped pipeline.**

**The Lean spec — `y = ln(I / (g·A))`, λ absorbed into `Fcal` — is a VALID UNIT-REDUCED FORM, NOT an error.** The reviewer's "fundamental error" flag is a **FALSE POSITIVE**.

The spec adopts a **photon-rate** (or `F`-absorbed) convention: `I = Fcal·A_k·n_k`. This is *structurally identical* to the canonical CF-LIBS forward model in **Tognoni et al. (2010)**: `I_ij = F·n_i^s·A_ij = F·(A_ij g_i/U^s(T))·n^s·exp(−E_i/kT)`, where `F` is the "experimental parameter accounting for optical collection efficiency, plasma number density, and plasma volume" — the same lumped `F` the original **Ciucci et al. (1999)** procedure uses. In that convention the ordinate is `ln(I/(g·A))` with **no explicit λ**, because λ has been folded into `F`. The spec is therefore internally self-consistent: it folds λ into `Fcal` in the forward map *and* drops λ from the ordinate, so `Fcal` cancels in the two-line slope regardless of its contents. Its theorems `boltzmann_plot_intensity` and `temperature_from_two_lines` are **logically sound and proof-valid as stated**.

**Unambiguous bottom line on physics errors: NEITHER repo contains a physics error.** The pipeline uses the measurement-standard energy ordinate; the spec uses a self-consistent photon-rate reduction. The *only* defensible criticism is **documentation**, not mathematics: the `ForwardMap.lean` docstring claims a **scalar** `Fcal : ℝ` "absorbs `hc/4π λ_ki`." That phrasing is *physically loose* because `λ_ki` is per-line, not line-independent — a single scalar cannot literally carry a per-line quantity. But the spec never co-plots distinct-λ lines against a λ-dependent `F`, so this looseness **falsifies no theorem**; it is a latent interpretation gap at the data interface, not a broken proof.

---

## 2. When the two conventions agree vs diverge — does λ matter for T and composition?

The two ordinates differ **line-by-line** by `ln(λ_ki) + const`. Decompose the difference: `hc` and `4π` are global constants (cancel in slope-differencing and in the `Σ C_s = 1` closure ratio), but **`λ_ki` is per-line and correlates with `E_k`** across a real species line set.

**They AGREE (λ is harmless) when:**
- λ is **constant** across all fitted lines, **OR**
- intensity is **truly photon-rate** (`I ∝ A_ki n_k`, no `hc/λ`), **OR**
- comparisons are restricted to lines sharing the same λ.

In these cases `hc/λ` is a single global constant → it cancels exactly in the slope (→ T) and in the closure (→ composition). A **uniform global λ-rescale** (e.g. every λ in metres instead of nm, consistently) is verified invariant in slope **and** intercept to 1e-12 — it is absorbed into `Fcal`/`F` and is observationally indistinguishable. This is an *intended, harmless blind spot*, consistent with the D2JA00241H ruling that a **constant** factor "without changing the slope … will shift its intersection with the y-axis."

**They DIVERGE (λ is load-bearing) when — the real CF-LIBS case — `λ_ki` VARIES across the multi-line fit:**

The per-line `−ln λ_ki` term does **not** cancel. Because λ correlates with `E_k` across a species' line set, dropping it **biases BOTH the slope (temperature) AND the intercept (composition)**, not just the intercept. The D2JA00241H "constant-factor shifts intercept only" rule does **not** apply here — that rule is explicitly about *constant* factors, whereas `λ_ki` is a per-line, `E_k`-correlated term.

**Quantification:**
- For a line set spanning **240 nm to 660 nm**, `ln(λ)` differs by `ln(660/240) ≈ 1.01` across the fit — a ~1.0 spread in the ordinate that maps directly onto a tilted slope.
- The numerical bridge experiment makes the slope bias concrete: a λ-drop bug yields **slope −0.689 vs the correct spec slope −0.419** — a gross temperature error, not a rounding effect.
- The error is **largest when the line set spans a wide wavelength range and λ correlates with E_k** — i.e. precisely the multi-element, wide-window CF-LIBS regime this codebase targets.

So: **for any real multi-line fit, the λ matters for both T and composition, and the energy ordinate (our pipeline) is mandatory.** The spec's reduced ordinate is correct only because it never feeds itself distinct-λ energy data.

---

## 3. The other three spec formulas — correct / reduced / error

| Spec formula | Location | Status | Basis |
|---|---|---|---|
| **Forward intensity** `I = Fcal·A_k·n_k` | `ForwardMap.lean:56` | **REDUCED** (not error) | Structurally identical to Tognoni 2010 `I_ij = F·(A_ij g_i/U)·n·exp(−E_i/kT)`. Same single root cause as the λ flag: photon-rate convention, λ folded into `Fcal`. Internally proof-valid; diverges from the measurement-standard `ln(Iλ/gA)` ordinate only at the data interface. Every prefactor (`hc`, `4π`, path length `L`, solid-angle/efficiency) **is** line-independent and legitimately collapsible; `λ_ki` is the **sole exception** because `hc/λ_ki = E_k − E_i` is energy-dependent. |
| **Saha factor** `S(T) = 2·(U_{z+1}/U_z)·(2π m_e k_B T/h²)^{3/2}·exp(−χ/k_B T)` | `Saha.lean:68` | **CORRECT** | Matches the Saha–Eggert equation in Griem, *Principles of Plasma Spectroscopy*, and the standard astrophysics/LIBS form `n_e·n_{z+1}/n_z = 2(U_{z+1}/U_z)(2π m_e k_B T/h²)^{3/2} exp(−χ/kT)`. Factor-by-factor: leading **2** = electron spin statistical weight `2g_e` (`g_e=1`); `(2π m_e k_B T/h²)^{3/2}` = inverse-cube thermal de Broglie wavelength, dimensionally `m⁻³` (correct number-density dimension); U-ratio orientation (higher stage in numerator) correct; `exp(−χ/kT)` sign correct (ionization costs energy → suppresses higher stage). The cleanest of the three — no missing or wrong factor. |
| **Self-absorption** `SA(τ) = (1 − e^{−τ})/τ` | `SelfAbsorption.lean:52` | **CORRECT**, and unusually well-derived | Standard **line-integrated homogeneous-slab curve-of-growth escape factor**: from radiative transfer `I = S(1−e^{−τ}) → Sτ` (thin), so dimming relative to thin emission is `(1−e^{−τ})/τ` (→1 as τ→0, →1/τ as τ→∞). Notably the spec does **not assume** SA — it defines `slabIntensity S τ = S(1−e^{−τ})` from radiative transfer and **proves** `slabIntensity_eq_thin_mul_SA`, i.e. SA is *derived* as the genuine emergent/thin ratio. Also proves SA ∈ (0,1], the downward composition-bias direction, and exact left-invertibility `I_thin = I_meas/SA(τ)`. Citations: Tatum *Stellar Atmospheres* Ch.11 (Curve of Growth); ADAS escape-factor formalism; Aragón & Aguilera (2008) growth-curve. **Caveat (modeling-fidelity note, not an error):** `(1−e^{−τ})/τ` is the line-integrated *uniform-slab, flat-line* escape factor — a homogeneous-plasma idealization, distinct from the wavelength-resolved `τ(λ)` Aguilera–Aragón / duplicating-mirror growth curve (whose core saturates as `√(ln τ)`, not `1/τ`). That is the textbook leading-order optically-thick correction, not a mistake. |

**Cross-cutting conclusion:** Of the four audited formulas, **only the forward-intensity λ-handling shares the flagged issue**, and even there it is an internally-consistent convention choice that keeps every verified theorem valid. Saha is correct; self-absorption is correct and well-derived. **No broken theorem exists in the spec.**

---

## 4. The fixes

### (a) Test bridge — `I_our = I_spec / λ` with **distinct** per-line λ — CONFIRMED CORRECT and λ-sensitive

To regression-test the pipeline (energy convention, `ln(Iλ/gA)`) against the spec's reduced fixtures (photon-rate, `ln(I/gA)`), convert via `I_our = I_spec / λ` with **distinct per-line λ**.

**Why it is exact (algebraic identity, not a tolerance match):**
```
ln(I_our·λ/(gA)) = ln((I_spec/λ)·λ/(gA)) = ln(I_spec/(gA))
```
The λ cancels point-by-point for **any** λ vector, so slope (→T), intercept (→N/U→concentration), and composition are reproduced **EXACTLY** (verified to 1e-12). `hc/4π` lives in `Fcal`/`F` and cancels in both slope and closure, so it never appears in the bridge.

**Why distinct λ is load-bearing (verified numerically):**
- **λ-drop bug** (pipeline forgets the λ factor): ordinate gains a λ-dependent `−ln λ` term that, since λ correlates with `E_k`, **tilts the slope** → wrong T. **CAUGHT** (slope −0.689 vs spec −0.419).
- **λ² bug** (`I·λ²/gA`): extra `+ln λ` → slope biased → **CAUGHT**.
- **per-line wrong-units bug** (nm vs m mismatched between intensity-conversion and ordinate): breaks per-line cancellation → **CAUGHT**.
- **`λ = 1` (uniform) fixtures are BLIND** to all of the above — every λ-bug cancels uniformly and the test passes spuriously. **Fixtures must carry distinct, realistic, E_k-correlated per-line λ — never λ=1.**

**Test-design refinements (both required):**
1. Assert against **slope/T, intercept, AND recovered composition** — not slope alone. A self-cancelling bug (divide input by λ *and* multiply ordinate by λ) is slope-invariant but is caught by the intercept/composition leg (already pinned by the spec's `closure`/`ols` fixtures).
2. Document the **uniform global λ-rescale as an intended, uncatchable blind spot** — it is absorbed into `Fcal` and cancels in closure (`Σ C_s = 1`); analytically harmless, not a test gap.

**Insertion point:** the oracle's `oracle/check_fixtures.py` IMPL block. Swap `recover_T`, `ols_density`, `classic_density` to call the pipeline with `I_our[k] = I_spec[k]/λ[k]`; add a per-element `"lambda"` array (distinct, E_k-correlated) to `oracle/fixtures.json` if absent; assert against the same fixture targets at `rtol 1e-6`.

### (b) Should the spec add an energy-intensity forward variant? — YES (lightweight, recommended)

The spec is **acceptable as-is for correctness** — its inverse theorems are dimensionless and the test bridge alone closes the loop. But adding a thin **energy-intensity sibling** is recommended because it (i) permanently removes the reviewer's false-positive by making `hc/4πλ` explicit and machine-proving the λ-form Boltzmann plot is the *same* affine line; (ii) is ~30 lines reusing `population` verbatim with **no new axioms** (pure `log`/`ring`, preserving `{propext, Classical.choice, Quot.sound}`); (iii) documents *in machine-checked form why the two conventions agree* — the exact artifact this review exists to produce. **Do not rewrite the core** `lineIntensity = Fcal·A·n` — every downstream theorem rests on it; add the energy form as a *sibling* (`ForwardMapEnergy.lean` or a new section) that reduces to it.

**Prompt-ready addendum for the cflibs-formal agent:**

> Add an energy-intensity forward variant that makes the `hc/4πλ` photon-energy factor explicit and proves the wavelength-form Boltzmann plot agrees with the existing reduced (Fcal-absorbed) one. This closes a reviewer flag that the spec's `ln(I/(g·A))` ordinate "omits λ" — it doesn't (λ is inside `Fcal`), but make it explicit and machine-checked. Keep the existing `lineIntensity = Fcal·A·n` as the canonical map; add the energy variant as a sibling that reduces to it. **No new axioms; pure `log`/`ring`.**
>
> ```lean
> /-- Energy-intensity forward map: I = (hc/(4π·λ))·A_k·n_k·Fgeo, with explicit
>     per-line wavelength λ : ι → ℝ and geometry/calibration Fgeo (NO λ inside). -/
> noncomputable def lineIntensityEnergy (hc fourPi : ℝ) (kB T N Fgeo : ℝ)
>     (g E A lam : ι → ℝ) (k : ι) : ℝ :=
>   (hc / (fourPi * lam k)) * A k * population kB T N g E k * Fgeo
>
> /-- Reduction: with Fcal := hc·Fgeo/(4π·λ_k) (PER-LINE), lineIntensityEnergy = lineIntensity. -/
> theorem lineIntensityEnergy_eq_lineIntensity
>     (hc fourPi kB T N Fgeo : ℝ) (g E A lam : ι → ℝ) (k : ι)
>     (hλ : 0 < lam k) (hfp : 0 < fourPi) :
>     lineIntensityEnergy hc fourPi kB T N Fgeo g E A lam k
>       = lineIntensity kB T N (hc * Fgeo / (fourPi * lam k)) g E A k := by
>   unfold lineIntensityEnergy lineIntensity; field_simp; ring
>
> /-- WAVELENGTH-FORM Boltzmann plot: ln(I·λ/(g·A)) is affine in E_k with slope −1/(kB·T),
>     intercept ln(hc·Fgeo·N/(4π·U)). The explicit λ cancels the 1/λ photon-energy factor —
>     the real-LIBS / energy form, matching the CF-LIBS-improved pipeline y = ln(I·λ/(g·A)). -/
> theorem boltzmann_plot_intensity_wavelength [Nonempty ι]
>     {hc fourPi kB T N Fgeo : ℝ} {g E A lam : ι → ℝ}
>     (hg : ∀ k, 0 < g k) (hN : 0 < N) (hhc : 0 < hc) (hfp : 0 < fourPi)
>     (hFgeo : 0 < Fgeo) (hA : ∀ k, 0 < A k) (hλ : ∀ k, 0 < lam k) (k : ι) :
>     Real.log (lineIntensityEnergy hc fourPi kB T N Fgeo g E A lam k * lam k / (g k * A k))
>       = Real.log (hc * Fgeo * N / (fourPi * partitionFunction kB T g E)) − E k / (kB * T)
> -- NOTE: λ does NOT cancel here (Fcal carries 1/λ). Expand directly: I·λ = (hc·Fgeo/4π)·A·n,
> -- then mirror boltzmann_plot_intensity (field_simp; Real.log_mul; Real.log_exp; ring).
>
> /-- Temperature from two lines, wavelength form: slope of ln(I·λ/(g·A)) vs E recovers
>     1/(kB·T) exactly; hc, Fgeo, λ, N, U, g, A all cancel. -/
> theorem temperature_from_two_lines_wavelength [Nonempty ι] (/- same hyps -/)
>     (i j : ι) (hE : E i ≠ E j) :
>     (Real.log (lineIntensityEnergy /-…-/ j * lam j / (g j * A j))
>        − Real.log (lineIntensityEnergy /-…-/ i * lam i / (g i * A i))) / (E i − E j)
>       = 1 / (kB * T) := by
>   rw [boltzmann_plot_intensity_wavelength /-…-/]; field_simp; ring
> ```
>
> Then: (1) update the `ForwardMap.lean` docstring and `CONTEXT.md` "Boltzmann plot" domain line to document **both** conventions, citing **Khelladi et al., EPJ AP 101 (2023) ap230072** (λ-form) and **Ciucci et al., Appl. Spectrosc. 53 (1999) 960** (F-absorbed form); delete/repair the loose claim that a *scalar* `Fcal` "absorbs `hc/4π λ_ki`" — state plainly that the per-line `λ_ki` is the load-bearing term that cancels only when λ is constant across the fitted line set. (2) Add an oracle fixture instantiating `temperature_from_two_lines_wavelength` with **distinct per-line λ** so the numerical bridge regression-tests against a *proven* λ-form theorem, not just the reduced one.

---

## Summary

| Item | Verdict |
|---|---|
| Pipeline `ln(Iλ/gA)` | **CORRECT** — literature standard (Aragón & Aguilera 2008; Alrebdi 2022; Khelladi/EPJ AP 2023; D2JA00241H 2022). No change. |
| Spec `ln(I/gA)` | **VALID UNIT-REDUCED FORM, NOT an error.** Reviewer's "fundamental error" flag is a **FALSE POSITIVE**. Photon-rate / F-absorbed convention (Ciucci 1999; Tognoni 2010); proof-valid. |
| Physics error? | **NEITHER repo.** Spec's only defect is a *loose docstring* (scalar Fcal "absorbs hc/4πλ_ki"), which breaks no theorem. |
| λ matters for T & composition? | **YES, for both**, whenever λ varies across the fit (the real multi-line case). ~1.0 ln(λ) spread over 240–660 nm; bias tilts slope (T) AND intercept (composition). Cancels only for constant-λ / photon-rate / uniform global rescale. |
| Saha factor | **CORRECT** (Saha–Eggert; Griem). |
| Self-absorption `(1−e^{−τ})/τ` | **CORRECT**, derived not assumed; uniform-slab idealization (documented scope). |
| Test bridge | `I_our = I_spec/λ`, **distinct E_k-correlated per-line λ (never λ=1)**, assert slope+intercept+composition. Exact (1e-12) and λ-bug-sensitive. |
| Spec energy variant | **Recommended**, lightweight (~30 lines, no new axioms). Prompt-ready addendum above. |

Relevant paths: `cflibs/inversion/physics/boltzmann.py` (lines 56, 123, 689, 1400), `cflibs/inversion/physics/boltzmann_jax.py:11`, `/home/brian/code/cflibs-formal/CflibsFormal/ForwardMap.lean`, `/home/brian/code/cflibs-formal/CflibsFormal/Saha.lean`, `/home/brian/code/cflibs-formal/CflibsFormal/SelfAbsorption.lean`, `/home/brian/code/cflibs-formal/oracle/check_fixtures.py`, `/home/brian/code/cflibs-formal/oracle/fixtures.json`.