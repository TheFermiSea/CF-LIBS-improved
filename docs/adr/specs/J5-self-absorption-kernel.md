# J5 Implementation Spec — Jittable Observable-Gated Self-Absorption

**Bead:** J5 · **ADR:** [ADR-0004](../ADR-0004-jittable-inversion-pipeline.md) §4 row 7 · **Track:** B · **Depends:** J0, J4 (masked-median helper + array conventions) · **Estimated effort:** 3–5 pd

## 1. Goals

Port `ObservableSelfAbsorptionCorrector.correct` (`cflibs/inversion/physics/self_absorption_observable.py:354-427`) into `jitpipe/selfabs.py`. It is applied ONCE pre-fit in `solve()` (`solve/iterative.py:1586-1590`) and never reads the composition (the audited F4 feedback loop was deleted) — so it is a pure `(B, L)` array transform.

## 2. Current algorithm and jit-breakers (file:line)

Ladder: (a) **doublet pass** (`self_absorption_observable.py:466-545`) over `find_doublet_pairs` (`physics/self_absorption.py:508+`; same species + same upper level within 1 meV), solving `f(τ₁)/f(τ₁/ρ) = r_meas/r_thin` with `scipy.optimize.brentq` on τ₁∈[1e-4, 30], xtol 1e-6 (`self_absorption.py:447`); same-sign-endpoints "observably thin" branch returns τ=1e-4 (`:424-444`); two-sided significance gate combining intensity + A_ki uncertainties with a 0.10 absolute floor (`_ratio_deviation_significant`, `self_absorption_observable.py:433-464`); τ > 5 validity ceiling → force-suspect, never boost (`:507-525`, constant at `:90`). (b) **Planck-ceiling** closed form `τ₀ = −ln(1 − I_peak/B_λ(T))` (`:115-172`) + Doppler curve-of-growth escape factor as a fixed 64-term alternating series (`:175-201`), validity τ ≤ 3 (`:84`). (c) **Suspect pass** (`:614-648`): per-element median intensity; Fayyaz low-E_i (<0.74 eV, `:95`) AND bright → σ×3 (down-weight w/9), never boosted.

Breakers: `brentq` (host root-finder); combinatorial pair discovery over a Python list; **order-dependent first-usable-pair-wins** (`:480`: skip if index already corrected/cleared); dict bookkeeping keyed by float wavelength (`:377`); `ValueError`-based validation (`self_absorption.py:430-446`); warning strings; `replace(...)` dataclass churn.

## 3. Redesign

- **Pair discovery is atomic-data-static:** which lines form doublets depends only on (element, stage, E_k, λ, g_k, A_ki) — none of it measured. Precomputed at snapshot build (J0): `(P, 2)` int32 line-index pairs + per-pair ρ and r_thin constants (`self_absorption.py:222-233`).
- **Root solve → fixed-K bisection:** residual is smooth and pre-bracketed [1e-4, 30]; 60 fixed bisection steps give a relative bracket ≪ brentq's xtol=1e-6 — branchless, vmapped over all P pairs at once. The same-sign thin test becomes a mask reproducing the τ=1e-4 branch exactly.
- **First-pair-wins → deterministic claim resolution:** sort pairs in the reference iteration order; per-line `claimed_by = scatter-min(pair_index)` over the pair→line incidence; only the claiming pair's correction applies. Order-free, deterministic; reproduces the sequential semantics whenever each line appears in ≤1 *usable* pair (the common case); the multi-claim contract is documented and fixed by the scatter-min rule.
- **Planck pass:** already pure math incl. the fixed 64-term series — port verbatim. **Suspect pass:** per-element masked median (segment median over E groups, J4 helper) + boolean signature; inflation = `where(suspect, σ·3, σ)`.
- **Output:** corrected `(B, L)` intensity/σ + per-line uint8 method code + per-line τ, feeding the same padded arrays J4 consumes; `ObservableSAResult` counters (n_corrected/n_suspect/max_tau, `:406-411`) as masked reductions; warning strings host-side from codes.

## 4. Tolerance contract

τ atol 1e-6 per pair (= brentq xtol); corrected intensities rtol 1e-6; **identical corrected/suspect/cleared sets** under the documented pair-priority contract; quality counters exact.

## 5. Acceptance criteria

1. Contract green on golden fixtures + ChemCam BHVO-2 line lists.
2. **Adversarial overlapping-doublet fixture** (a triplet sharing one upper level → one line in two usable pairs) with the claim-resolution outcome asserted and documented.
3. Significance gate, τ-ceiling force-suspect, and never-boost invariants property-tested (corrected intensity ≤ boosted bound; suspects only inflate σ).
4. jit + vmap (B=16) + padding invariance; grad-finite hard assert (the escape-factor term feeds J7's COG-in-regression extension).

## 6. Test plan

`tests/jitpipe/test_parity_selfabs.py`: A/B vs `ObservableSelfAbsorptionCorrector` on identical observation arrays; bisection-vs-brentq property test over randomized (ρ, r_meas/r_thin) grids; the triplet fixture; counter equality.

## 7. Risks

LOW–MED. The only subtlety is the claim-ordering contract (mitigated by the adversarial fixture). Physics is closed-form throughout.

## 8. Dependencies / files

Depends J0 (snapshot pairs), J4 (helpers). Enables J7 (corrected arrays + escape-factor term). Files: `cflibs/jitpipe/selfabs.py`, tests. Reference untouched.
