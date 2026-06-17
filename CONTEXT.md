# CONTEXT — CF-LIBS domain glossary

Shared vocabulary for the CF-LIBS pipeline. Architecture reviews and design
docs should use these terms exactly. (Seeded 2026-06-03 by the architecture
review while deepening the partition-function provider seam.)

## Plasma & atomic data

- **Plasma state** — `(T, n_e, composition)`. The forward model maps a plasma
  state to a synthetic spectrum; inversion recovers it from a measured one.
- **U(T) / partition function** — the per-species statistical sum
  `U(T) = Σ_i g_i exp(-E_i / kT)`. It is the denominator of every Boltzmann
  population and scales every Saha ionization ratio, so a wrong `U(T)` biases
  composition on whatever path consumes it.
- **Direct-sum `U(T)`** — `U(T)` computed by summing `g_i exp(-E_i/kT)` over the
  species' tabulated **energy levels**. Preferred when levels are present; it is
  the reference the validation gate trusts.
- **Polynomial `U(T)`** — `ln U = Σ aₙ (ln T)ⁿ` (Irwin form), a fitted fallback
  used when energy levels are absent or when a vmap-traced array form is needed.
  Only valid inside its fit window `[t_min, t_max]` and must be floored at the
  ground-state weight `g0`; extrapolating it unguarded produces order-of-magnitude
  errors (the defect class behind the 2026-06-03 audit).

## The partition-function provider (the seam)

- **Partition-function provider** — the single seam for obtaining `U(T)` for a
  species. Its **policy**: *prefer direct-sum when energy levels exist; otherwise
  the polynomial fallback; and always clamp `T` to `[t_min, t_max]` and floor at
  `g0`.* The policy lives in one factory (`AtomicDatabase.partition_function_for`)
  so the decision is made once, never re-implemented at call sites.
- **CPU scalar adapter** — the provider form callers use at Python runtime
  (`.at(T)` on a scalar / numpy array). Used by the default `invert` / `analyze`
  / iterative / closed-form solvers.
- **JAX batched adapter** — the provider form for jit/vmap kernels: the same
  direct-sum-fit coefficients + `[t_min, t_max]` + `g0` baked into static arrays
  inside the **atomic snapshot** at build time, evaluated by the one shared
  *guarded* `polynomial_partition_function_jax`. Used by the manifold and
  Bayesian forward models. (Direct-sum-preferred is a build-time choice here,
  because vmap needs static, fixed-shape arrays.)
- **Atomic snapshot** — the immutable, jit-friendly bundle of atomic arrays
  (lines, levels, partition coefficients + bounds) the JAX forward models consume.

Both adapters are produced by the one factory from the **same** coefficients and
bounds; that is what makes "direct-sum-preferred, always guarded" provable in a
single place rather than smeared across nine call sites.

## Identification

- **Identifier** — a module that decides which elements are present in a
  spectrum (ALIAS, comb, correlation, spectral-NNLS, hybrid_union). Each stacks
  its own **detection gates**; the gates are genuinely different per identifier
  (a unified decision abstraction was rejected by the deletion test — see
  bead n9lz for the observability follow-up).
- **Detection gate** — a per-element pass/reject test inside an identifier. A
  gate must be **count-invariant**: its effective bar must not tighten as the
  candidate-element count grows (see the 2026-06-03 candidate-count-fragility
  audit; the `coeff / Σ-over-candidates` form is the anti-pattern).

## Scoring (the confusion seam)

- **Confusion rule** — the single mapping from `(truth, predicted, don't-care
  band)` to a per-element label `tp | fp | fn | tn` (or *skip*). It lives in one
  place, `cflibs/benchmark/scoring.py` (`classify_element`), so the
  synthetic-corpus benchmark and the observability per-element aggregator cannot
  drift on the semantics. Callers differ in **scope** (panel-based with a TN
  cell vs set-based over the elements that appear), never in **rule**.
- **Don't-care band** — the per-spectrum set of real-but-sub-detection-floor
  trace elements (`0 < fraction < presence_threshold`). Detecting or missing a
  don't-care element is neither rewarded (TP) nor penalised (FP), and never an
  FN — the confusion rule returns *skip* for it. Empty at the legacy
  `presence_threshold` (1e-4); non-empty only when a meaningful floor is set.
- **Scoring panel** — the candidate-element set a row's confusion is computed
  over. The full panel and the **ever-present** panel (truth ∩ candidates) are
  both always reported; restricting to ever-present never penalises the
  identifier for a candidate that never occurs in truth.
- **Scoring row** (`ScoringRow`) — the typed confusion core of one
  spectrum × algorithm benchmark record (its three element sets), built from the
  wire-format row dict via `ScoringRow.from_row`. The dict stays the persisted
  form (`per_spectrum.jsonl` / Parquet); the type carries the rule on the
  in-memory compute path.
