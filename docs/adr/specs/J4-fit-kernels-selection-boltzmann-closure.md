# J4 Implementation Spec — Fit Kernels: Line Selection, Boltzmann/SB-Graph, Closure

**Bead:** J4 · **ADR:** [ADR-0004](../ADR-0004-jittable-inversion-pipeline.md) §4 rows 5/6/9 · **Track:** B (back-end) · **Depends:** J0 · **Estimated effort:** 5–9 pd (selection 1–2, Boltzmann + SB-graph 3–5 incl. the Schur-equivalence property proof, closure 1–2)

## 1. Goals

Implement `jitpipe/fit.py`: the three pure-algebra back-end stages as batched masked kernels over `(B, E, N)` padded arrays (E ≤ E_max=32 elements, N ≤ 20 lines/element after selection per `physics/line_selection.py:146`). Reuse the existing bit-exact JAX twins verbatim where they exist; prove and exploit the SB-graph ≡ common-slope identity so **no general lstsq is needed on device**.

## 2. Line selection (`cflibs/inversion/physics/line_selection.py`)

**Reference:** `LineSelector.select` (`:176-239`): score = SNR × (1/σ_atomic) × isolation (`:322-360`); isolation `1 − exp(−Δλ_min/0.1 nm)` from an O(L²) Python loop (`:362-389`); sequential gates (`:241-264`): SNR < min_snr; resonance (default off, `:118-121,143`); isolation < 0.5; per-element stable sort + top-K=20 truncation (`:266-290`); energy-spread/min-lines are warning strings only (`:292-320`). Breakers: variable-length lists, dict grouping (`:217-219`), Python sort whose stable-tie order feeds truncation (`:276`), reason strings, resonance-key sets (`:337,344`).

**Redesign:** pure mask transform on `(B, L)` arrays (wl, I, σ_I, E_k, element-idx, resonance, σ_atomic from snapshot). Isolation = masked pairwise |Δλ| min (L is a few hundred) or sort-based neighbour diff. Gates = boolean AND. Per-element top-K via `jax.lax.top_k` with static K=20 producing a fixed `(B, E, N)` selection mask consumed directly by the fit (the list form is never materialised). Reasons → uint8 codes decoded host-side; energy-spread/min-lines → numeric per-element diagnostics (`spread_ev (B,E)`, `n_valid (B,E)`) thresholded host-side.

**Contract:** exact selected-set equality for tie-free inputs; deterministic tiebreak for ties = lower original index wins (matches Python stable sort), encoded by lexicographic packing (score, −index); scores rtol 1e-12.

## 3. Boltzmann fitters + SB-graph

**Reference layers:** (a) per-element robust fitters, `physics/boltzmann.py`: sigma-clip (`:326-413`, ≤10 iterations, mask shrinkage `:347-349`, early breaks `:387-393`); `_fit_sigma_clip_jax` (`:415-535`) already delegates the inner WLS to the fully-jitted `batched_boltzmann_fit` (`physics/boltzmann_jax.py:111-200`) — the right inner kernel; RANSAC (`:630-732`, `default_rng(42)`, 100 sequential trials, MAD threshold `:657-667`, degenerate-sample `continue` `:681-687`); Huber IRLS (`:734-842`, convergence break `:796`); multiplet aggregation (`:205-324`). (b) The default pooled per-element-centered common-slope plane `_fit_common_boltzmann_plane` (`solve/iterative.py:1349-1452`) with weight cap `_cap_boltzmann_weights` (`:2911-2944`, clip at K×masked-median) — **bit-exact JAX twins exist**: `_common_slope_kernel` (`:2698-2773`), `_saha_correct_kernel` (`:2675-2696`). (c) The *production* estimator (geological preset): pooled SB-graph `_fit_saha_boltzmann_graph` (`:1454-1539`) — ion-shifted rows with **unit weights by design** (`:2776-2815`; the docstring records that inverse-variance weighting re-creates Fe over-attribution) solved via `np.linalg.lstsq` on a dynamic `(N_rows, 1+E)` design (`:2818-2886`).

**Redesign:**
- sigma-clip → fixed-K=10 `lax.scan` of masked reweighting (weight × inlier indicator `|r| ≤ 2.5·σ_r(masked)`), carrying fixed-shape `(w, mask)`; CPU early termination = idempotent fixed point ⇒ identical result; the rare `std_res==0` break becomes a `where`. Inner solve = `batched_boltzmann_fit`.
- Huber → fixed-K IRLS with sort-based masked-MAD scale; 1e-8 break = fixed-point idempotence.
- RANSAC → vectorized trials: pre-draw all `(T, 2)` index pairs with counter-based `jax.random`; evaluate in parallel; argmax inliers with first-max tie = lowest trial index (matches the strictly-greater sequential update `:694-696`). **RNG streams cannot match `np.random.default_rng(42)` bit-for-bit — contract is fixed-seed self-regression + statistical parity.**
- Common-slope plane: reuse `_common_slope_kernel` verbatim with one extra vmap axis → `(B, E, N)`.
- **SB-graph without lstsq:** with unit weights the `(1+E)×(1+E)` normal matrix is arrow-shaped (dense slope column, diagonal intercept block); the Schur complement reduces it to slope = Σ_s Σ_l (x−x̄_s)(y−ȳ_s) / Σ_s Σ_l (x−x̄_s)², q_s = ȳ_s − m·x̄_s — i.e. **the SB-graph with unit weights IS the centered common-slope kernel with w=1 on ion-shifted coordinates** (independently confirmed by audit Q2, `docs/audit/2026-06-09-overhaul/02-inversion-solver.md:378`). Implement as `_saha_correct_kernel` → `_common_slope_kernel`(w=mask) + closed-form slope-variance (`inv(AᵀA)·σ²` equivalent, cf. `:2873-2883`) and global R² from the same sums. Validity screen (`:2803`: A_ki≤0 | g_k≤0 | I≤0) → mask; under-determined `None` (n_rows < 1+E, `:1524-1525,2844-2846`) → per-spectrum validity flag.
- Masked median helper (shared with weight cap, J5, J6): sort with +inf fill, gather `(n−1)//2` and `n//2`, average — exact `np.median` for both parities.

**Contract:** SB-graph slope/intercepts rtol 1e-10 vs `np.linalg.lstsq` on identical rows (linear-algebra identity; property-tested over randomized fixtures **including collinear-x and all-equal-weight degenerate cases**); T rtol 1e-8; sigma-clip final inlier mask bit-equal when no residual is within 1e-9·σ of the boundary, slope rtol 1e-8 (existing precedent `boltzmann.py:433-437`); Huber rtol 1e-8; RANSAC self-regression only.

## 4. Closure (six modes) + keystone gate

**Reference:** `physics/closure.py` — standard (`:537-593`), matrix w/ missing-element fallback (`:595-660,629-635`), oxide with `OXIDE_OXYGEN_PER_CATION` (`:662-727`, table `:62-73`, default 1.0 `:712`), ILR (`:729-801`) — **an identity round-trip** (audit: "they change conditioning, not physics", `02-inversion-solver.md:364-365`) differing from standard only via the `LOGRATIO_CLIP_FLOOR=1e-10` clip (`:34,158`) and `sorted(intercepts)` ordering (`:769`); PWLR adaptive-ridge (`:803-890,384-418`); Dirichlet-residual branch (`:892-1033,1004-1019`); keystone gate `validate_degeneracy` (`:495-535`). In the lax path standard/matrix/oxide are already pure JAX (`solve/iterative.py:567-612`); **ilr/pwlr/dirichlet go through `jax.pure_callback`** (`:614-655`) — serial under vmap, non-differentiable, and ILR is the constructor default (`:1046-1050`).

**Redesign:** masked linear algebra over `(B, E)`: lift the existing lax closures one batch axis; oxide factors = `(E,)` snapshot vector; missing-U elements = mask zeros (≡ the `continue` at `:570-571,704-705`); matrix-element-missing resolves statically at array build (as `:579-587` already does). **ILR implemented as standard closure** with the 1e-10 clip for exact parity; keep the explicit Helmert algebra (`:117-140`, static `(E, E−1)` matrix) available because J7 parameterises composition in ILR coordinates. PWLR closed-form (target = Ψ·log, λ from masked min, diagonal shrink, inverse via Ψᵀ + normalize; pivot argmax is a dynamic gather — jit-fine). Dirichlet = `jnp.where` over branches. Keystone gate in-body: `(max(C) > 0.8) & (n_valid ≥ 4)` — closes lax seam (iv) of ADR-0004 §4 row 10.

**Contract:** standard/matrix/oxide rtol 1e-12 vs the dict implementation; ilr/pwlr rtol 1e-10 + a documented ILR≡standard equivalence property on strictly-positive compositions; degeneracy flag exact.

## 5. Acceptance criteria

1. All contracts above green via direct A/B vs the reference functions on randomized + scoreboard-derived fixtures.
2. Schur-equivalence property test: 1,000 random (rows, E≤20) systems, jit SB-graph vs `np.linalg.lstsq`, rtol 1e-10, including degenerate fixtures (collinear x; single-element; n_rows = 1+E exactly; n_rows < 1+E → validity flag).
3. jit + vmap (B=16) + padding invariance + grad-finite (hard assert — these kernels sit inside J7's differentiated region).
4. Selection tie property test vs Python stable sort.
5. No `pure_callback` anywhere in `jitpipe/fit.py`.

## 6. Test plan

`tests/jitpipe/test_parity_fit.py` — three test classes (selection / boltzmann / closure), fixture style of `test_iterative_lax.py:79-121`; adversarial degenerate fixtures explicitly listed in AC-2.

## 7. Risks

LOW–MED: silent divergence on degenerate fixtures (collinear x, equal weights) — covered by AC-2; RANSAC RNG non-parity is by construction and documented; ILR clip/order parity is a one-line trap covered by the equivalence property.

## 8. Dependencies / files

Depends J0. Enables J5 (masked-median helper + array conventions), J7 (all kernels). Files: `cflibs/jitpipe/fit.py`, tests. Reference untouched.
