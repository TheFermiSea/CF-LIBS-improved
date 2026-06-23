# CF-LIBS Composition-Estimator Routing Policy

**Scope.** Which composition / plasma-parameter estimator the inversion pipeline runs
in which physical regime, with **each routing decision pinned to the proven theorem it
rests on**. The theorems live in the verified Lean spec (`~/code/cflibs-formal`,
`CflibsFormal/…`); their float mirrors are the reference formulas in
`oracle/check_fixtures.py` and the regression test `tests/oracle/test_spec_regression.py`.
The runtime gates this policy describes are implemented in
`cflibs/inversion/physics/identifiability.py` (routing preconditions) and
`cflibs/inversion/physics/reliability.py` (within-regime line ranking), built in
parallel with this document.

This is a **policy** document: it says what to run and why it is licensed, not how the
estimators are coded. The estimator implementations are
`BoltzmannPlotFitter` (`physics/boltzmann.py`), `ClosureEquation`
(`physics/closure.py`), `_escape_factor` (`physics/self_absorption.py`),
`SahaBoltzmannSolver` (`plasma/saha_boltzmann.py`), and `stark_ne` (`physics/stark_ne.py`).

---

## 0. The one prune that must happen first

**C-σ is not a second method. It is the classic inverse in different packaging.**

> `Alt.CSigma.csigmaComposition_eq_classicComposition` — *unconditional* identity:
> for all positive intensities, `csigmaComposition = Classic.classicComposition`. The
> per-species density inverses coincide
> (`csigmaDensity_offset_eq_classicDensity`); the "Saha-Boltzmann C_σ" master-line
> estimator and the classic per-species estimator are the **same algebraic
> left-inverse** in two log/exp packagings, not two independent procedures.

**Routing consequence — mandatory:**

- **Do NOT treat classic and C-σ as two methods.** Do **not** count them as two votes,
  do **not** average them, do **not** report "two estimators agree" — their agreement
  is a definitional identity (`csigma_agrees_classic` is just that identity applied to
  a forward spectrum), so agreement carries **zero** independent corroboration.
- **Keep exactly one.** Choose the packaging by *numerical conditioning only*: the C-σ
  master-line offset form (`csigmaOffset_of_lineIntensity`) avoids one division and is
  preferable when per-species partition-function ratios are ill-scaled; otherwise the
  classic form is fine. This is a floating-point choice, never a physics choice.
- The genuinely independent same-spectrum cross-check is **OLS vs classic**
  (`Alt.leastSquares_agrees_classic`), which is the *only* pair that differs off the
  noise-free fixpoint and therefore the only pair whose agreement is evidence (§2).

---

## 1. Decision table — regime → estimator → theorem

| # | Regime (routing precondition) | Estimator | Licensing theorem(s) | Module |
|---|---|---|---|---|
| R1 | **Optically thin, few clean lines.** Per species ≥ 1 isolated line, τ ≈ 0, low noise. | **Classic / C-σ** (single inverse; §0) | `Classic.classic_sound`, `classic_sound_sum_one`, `classicDensity_recovers`; `csigmaComposition_eq_classicComposition` collapses C-σ into it | `closure.py` `ClosureEquation.apply_standard`; `csigma.py` |
| R2 | **Optically thin, MANY noisy lines** per species, with **energy spread** (`Σ(E−Ē)² > 0`). | **OLS Boltzmann plot** | `Alt.leastSquares_sound`, `olsDensity_recovers`, `olsIntercept_of_forward`; noise-robust slope via `olsSlope_noise_gain` | `boltzmann.py` `BoltzmannPlotFitter.fit` |
| R3 | **Self-absorbed / optically thick**, τ **known** (or fit from a line pair). | **Classic ∘ SA(τ)-correction** | `Alt.selfAbsorbed_eq_classic_corrected`, `selfAbsorbed_sound`, `selfAbsorbed_corrects_bias`; thin limit `selfAbsorbed_eq_classic_thin` | `self_absorption.py` `_escape_factor` → `closure.py` |
| R4 | **Optically thick, τ unknown, but ≥ 2 lines** of differing width/strength (curve-of-growth pair). | **CoG ratio inversion** (recover τ, then R3) | `cogRatio_strictAntiOn`, `cogRatio_eq_intensity_ratio`, `cogRatio_injOn` (ratio is strictly monotone ⇒ invertible) | `self_absorption_observable.py`, `self_absorption_inputs.py` |
| R5 | **Electron density** `n_e` from an ion/neutral **stage ratio** (any thin regime above). | **Saha inversion** `n_e = S(T)/R` | `Saha.electronDensity_antitone`, `saha_relation`, `electron_density_identifiability` | `saha_boltzmann.py` `SahaBoltzmannSolver` |
| R6 | **`n_e` reliability / LTE gate** — Stark width and Saha ratio both available. | **Cross-check + McWhirter floor** | `StarkBroadening.stark_saha_lte_consistent`, `mcWhirterBound_mono_T`, `mcWhirterBound_mono_dE` | `stark_ne.py` + `saha_boltzmann.py` |
| X | **Single thick line, τ unknown** | **REFUSE / flag — not identifiable** | `selfAbsorption_breaks_identifiability` (counterexample: two `(N,τ)` give one intensity) | identifiability gate |

---

## 2. Thin regime — classic/C-σ vs OLS (R1 vs R2)

Both estimators are sound left-inverses of the thin forward model, so on **noise-free**
data they return the same composition:

> `Alt.leastSquares_agrees_classic` — fed the same forward intensities, OLS and classic
> return the identical composition (both equal the ground truth via `leastSquares_sound`
> and `classic_sound`).

**This is a noise-free corollary, not a routing equivalence.** Off the noise-free
fixpoint the two **disagree**, and that disagreement is the whole point of having both:

- **R1 (classic/C-σ):** uses one designated line per species. Minimal data; correct when
  that line is clean and isolated. Inverts via `classicDensity_recovers`.
- **R2 (OLS):** the **noise-robust** variant. With many noisy lines spanning an energy
  range, the least-squares slope down-weights per-line noise; its slope error scales as
  the inverse of the energy-spread (`olsSlope_noise_gain`, and the reliability ranking of
  §4). Route here whenever a species offers several lines with real `E_k` spread.

**Routing rule.** Per species: if only 1–2 clean lines → R1. If ≥ 3 lines with
`Σ(E_k − Ē)² > 0` → R2. Because they agree only in the noise-free limit, **report OLS
vs classic agreement as a data-quality signal** (small gap ⇒ low noise / valid thin
assumption; large gap ⇒ investigate self-absorption or line misidentification). This is
the only legitimate two-estimator vote in the pipeline (contrast §0).

---

## 3. Thick regime and the identifiability wall (R3, R4, X)

Self-absorption multiplies the thin intensity by the escape factor
`SA(τ) = (1 − e^{−τ})/τ ∈ (0,1]`. The corrected estimator is literally "classic, then
divide out SA(τ)":

> `Alt.selfAbsorbed_eq_classic_corrected` — `selfAbsorbedDensity = classicDensity` of the
> SA-corrected intensity. So **R3 = classic ∘ SA(τ)-correction**, and it inherits
> classic's soundness (`selfAbsorbed_sound`, `selfAbsorbed_corrects_bias`). In the thin
> limit `selfAbsorbed_eq_classic_thin` shows it degrades gracefully to R1.

**The wall (regime X).** A single thick line with unknown τ is **not identifiable**:

> `selfAbsorption_breaks_identifiability` — there exist `N₁ ≠ N₂` and `τ₁, τ₂ ≥ 0` with
> `selfAbsorbedIntensity(N₁,τ₁) = selfAbsorbedIntensity(N₂,τ₂)`. One measured intensity
> does not pin down `(N, τ)`.

**Routing consequence:** never run R3 with a guessed τ on a single line and report a
number. Either (a) τ is supplied/measured independently → R3; or (b) recover τ from a
**multi-line ratio** → R4. The CoG ratio is strictly monotone in column density, hence
invertible:

> `cogRatio_strictAntiOn` + `cogRatio_injOn` — the ratio of two curve-of-growth line
> intensities is strictly antitone (and injective) in `n`, so a measured ratio inverts to
> a unique τ. `cogRatio_eq_intensity_ratio` ties the abstract ratio to the observable
> intensity ratio.

Resolved τ then feeds R3. If neither (a) nor (b) is available, the routing gate returns
**X: refuse/flag** — this is the M7 refuse-to-report path, licensed directly by
`selfAbsorption_breaks_identifiability`.

---

## 4. Electron density and the LTE reliability gate (R5, R6)

**R5 — Saha path.** With a recovered ion/neutral stage ratio `R = n_{z+1}/n_z`,

> `Saha.electronDensity_antitone` + `saha_relation` — `n_e = S(T)/R` is the unique
> (strictly antitone in `R`) inverse of the Saha law `R·n_e = S(T)`. Implemented by
> `SahaBoltzmannSolver.solve_ionization_balance` / `electron_density_from_ratio`.

**R6 — reliability downgrade.** Two *independent* `n_e` diagnostics exist: Stark width
and the Saha stage ratio. They feed **different observations** (a WIDTH vs a RATIO), so
their equality is empirical evidence, not an identity:

> `StarkBroadening.stark_saha_lte_consistent` — *if* the Stark-width `n_e` and the
> Saha-ratio `n_e` agree **and** the result clears the McWhirter LTE bound, then a single
> consistent `n_e` exists satisfying both forward laws. The agreement hypothesis is
> explicitly *assumed*, not proven — that is exactly why it is a runtime gate.

**Routing rule (the gate):**

1. Compute `n_e^Stark` (`stark_ne.py`) and `n_e^Saha` (R5).
2. **Disagreement** between them ⇒ the LTE/single-zone assumption is suspect ⇒
   **downgrade / flag** the composition (do not certify `overall_reliable`).
3. **Below McWhirter** `n_e < 1.6e12·√T·(ΔE)³` (`constants.MCWHIRTER`, monotone in `T`
   and `ΔE` per `mcWhirterBound_mono_T/_dE`) ⇒ LTE invalid ⇒ **flag**. Use the
   resonance-line ΔE convention (see Serena memory `reference_mcwhirter_delta_e_physics`).
4. Both agree **and** clear McWhirter ⇒ `stark_saha_lte_consistent` licenses a single
   trusted `n_e`.

---

## 5. How the two parallel modules wire in

The decision table is gated by two orthogonal layers, both pinned to spec theorems:

**Identifiability preconditions (the routing gate) — `physics/identifiability.py`.**
Step-3 well-posedness theorems decide *whether any estimator may run at all* in a regime:

- thin composition: `compositionIdentifiable`, `density_identifiability`,
  `singleZone_identifiable`;
- thick: `thick_composition_identifiability` (preserved side) vs
  `selfAbsorption_breaks_identifiability` (lost side → regime X);
- `n_e`: `electron_density_identifiability`, `saha_joint_identifiability`.

If the precondition fails, route to **X (refuse/flag)** — accuracy-changing results
through a non-identifiable gate are forbidden (cflibs-verification-gate).

**Reliability ranking (within-regime line preference) — `physics/reliability.py`.**
Once a regime is chosen, the Step-4 conditioning result ranks *which lines to prefer*:

> `ErrorBudget.temp_rel_error_eq` — *exact* identity `|ΔT|/T = k_B·T̂·|Δβ|`, and
> `temp_rel_error_le` propagates a slope-error bound. For a two-line temperature the
> slope (and hence `n`, density) error scales as **`2/|ΔE|`** in the energy spread: wider
> `|E_i − E_j|` ⇒ better-conditioned `T`. Across a multi-line fit the OLS noise gain
> (`olsSlope_noise_gain`, `olsSlope_stable_l1/_l2`) is the many-line generalization.

So reliability **does not change which estimator runs** — it changes which lines that
estimator is fed: prefer line pairs/sets with the largest energy spread `|ΔE|` and the
best per-line SNR, exactly as the error-budget thresholds in
`physics/derived_thresholds.py` (mirroring `requiredEnergySpread_sufficient`,
`maxPerLineError_sufficient`, `requiredMinLinesStat`) quantify.

---

## 6. Summary rationale (prose)

The routing collapses to three questions answered in order:

1. **Is the result identifiable here?** (identifiability.py / Step-3 gate.) If a single
   thick line with unknown τ, or a failed Saha/composition well-posedness check — **stop
   and flag** (`selfAbsorption_breaks_identifiability` and siblings). This is the
   refuse-to-report wall.
2. **Which regime?** Thin+few → classic/C-σ (R1, one method — §0 prune).
   Thin+many+noisy → OLS (R2, the noise-robust variant that genuinely differs off the
   noise-free fixpoint). Thick → SA-correction (R3) if τ known, else CoG ratio (R4) to
   get τ first. `n_e` → Saha (R5).
3. **Is it reliable enough to certify?** (reliability.py / Step-4 ranking + R6 gate.)
   Prefer large-`|ΔE|`, high-SNR lines (conditioning `2/|ΔE|`); cross-check Stark vs Saha
   `n_e` and clear McWhirter, else downgrade.

The recurring discipline: **count independent evidence, not repackaged identities.**
C-σ ≡ classic is one method (§0); OLS-vs-classic and Stark-vs-Saha are the two real
cross-checks, and each is a *gate*, not a vote that can be assumed to pass.
