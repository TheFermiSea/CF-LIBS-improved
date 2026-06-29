The next ADR number is ADR-0007. I have everything I need. Here is the synthesized recommendation.

---

# ADR-0007 Investigation: SOTA Atomic-Data Lookup Representation for Real-Time CF-LIBS

*Synthesis of three independent investigations (hot-path profiling, framework benchmark, access-pattern design). All numbers measured on `ASD_da/libs_production.db` — 28,727 lines, 158 species, `idx_main(element,sp_num)`, CPU — with VALD/Kurucz-scale projections. No shipped code was modified; all benchmarks ran from isolated scratch venvs/scripts.*

---

## 1. Bottleneck verdict — where the SQLite latency actually is

**The bottleneck is not SQLite, the query planner, or the connection pool. It is per-call Python object materialization that runs on every call — including warm cache hits.** All three investigations independently converge on this.

Two distinct seams dominate the hot path:

| Seam | Where | Measured cost | What it actually is |
|---|---|---:|---|
| **`get_transitions()` COLD** | line-window lookup feeding `snapshot()` + every identify wavelength window | **187–203 ms** for Fe I (2439 lines); 28 ms via the production pandas path | **`pandas.iterrows()` + per-row `Transition` dataclass construction = 92% of cold cost.** Raw `sqlite execute+fetchall` is only **7.2 ms**; pool acquire/release is **4.3 µs** (negligible — pooling works). |
| **`partition_function_for(el,stage).at(T)`** | `iterative.py` `_evaluate_partition_functions`, **per iteration × N_el × 2 stages** | **173 µs** (Fe I) / **254 µs** (Fe II) even fully warm | `_spec_cache` already makes the SQL/fit compute-once (0.5 µs warm). The cost is pure Python churn that runs *anyway*: `to_provider()` rebuilds a frozen-dataclass provider + numpy→tuple conversion (**82 µs**) and `.at(T)` re-runs the exact Σgᵢexp(−Eᵢ/kT) direct-sum over hundreds of levels at Python runtime (**60 µs**). |

**How much it matters (measured, end-to-end):**

- A 10-element synthetic inversion total `solve()` = **13.6 ms**, of which `partition_function_for` = **3.82 ms (29%)** — and that run converged in only **4 iterations**. At the 20-iteration cap, partition lookups scale ~5× and dominate the solve.
- The "warm LRU = 6 µs" number is **misleading for the identify stage**: 50 *distinct* Fe I wavelength windows (one per detected peak — the realistic pattern) ran at **5,591 µs/call with a 0.00 hit rate**, because the LRU is keyed on exact `(element, stage, wl_min, wl_max)`. Every new window pays the full cold materialization.
- `get_ionization_potential` (6 µs warm) and the connection pool are **not** problems — they confirm the existing cache/pool layer is already fast for what it caches. `get_stark_parameters` (355 µs, uncached, unindexed `ABS()` scan) is a latent per-line forward-Stark cost but is not in the iterative loop today.

**Bottom line:** the storage engine is fine; the inner loop pays for (a) rebuilding Python objects the cache should have frozen, and (b) re-running the direct-sum partition evaluation at Python speed. A faster *database* fixes neither.

---

## 2. Recommendation — in-memory columnar NumPy

**Adopt an in-RAM columnar NumPy representation for the inference tier; keep SQLite as the offline build/source-of-truth.** This is the unanimous recommendation across all three investigations.

The static `lines` table is loaded **once at startup** into a dict keyed by species:

```
dict[(element, sp_num)] -> (wl_sorted: np.ndarray, cols: np.ndarray[N, k])
dict[(element, sp_num)] -> partition_coeffs   # (a0..a4, t_min, t_max)
dict[(element, sp_num)] -> ip_ev, mass, levels
```

Lookups become: **dict hash → `np.searchsorted` on the pre-sorted wavelength → zero-copy slice**, returning contiguous arrays. The hot path never touches SQLite and never builds a `Transition` object.

### Measured latency delta vs SQLite

| Operation | SQLite (production/raw) | In-RAM columnar | Speedup |
|---|---:|---:|---:|
| Line-window lookup (full species) | 676 µs raw / 28 ms cold prod | **0.14 µs** | ~140× vs raw, ~6,000× vs prod |
| Line-window sub-window (`searchsorted`) | 676 µs raw | **4.69 µs** | ~140× |
| Point lookup, 20k-window benchmark | 153 µs (indexed `:memory:`) | **3.23 µs** | ~47× |
| IP lookup | 5.6 µs | **0.13 µs** | ~43× |
| Partition U(T) eval | 6.9 µs fetch + 60 µs direct-sum | **1.69 µs** (coeffs in RAM) / **1.4 µs** (vectorised matrix) | ~100× vs direct-sum |
| Partition coeffs (Q2) | 7.53 µs | **0.086 µs** | ~88× |

The win is **flat to millions of lines** — lookup cost is hash + binary search, independent of table size. At 2M lines NumPy stays **~4 µs/lookup** while indexed SQLite climbs to **~14 ms/lookup**.

### Why NumPy, and when each alternative applies

The decisive comparison (28k-line reference DB, 20k windowed lookups, retrieval kernel):

| Representation | µs/lookup | Footprint | Use when |
|---|---:|---|---|
| **NumPy dict + `searchsorted`** | **3.23** | 13.6 MB RSS | **Inference inner loop — WINNER.** Hash + 2 binary searches + zero-copy slice, no query planner. |
| NumPy dict + boolean mask | 11.0 | (same) | Avoid — per-call mask alloc 3.4× slower than `searchsorted`. |
| SQLite `:memory:` + index | 153 | 18.3 MB | Build/storage only. ~47× slower on point lookups (per-row tuple fetch). |
| Polars per-species `.filter()` | 184 | 33.5 MB | **No** for inner loop — ~180 µs expr/dispatch overhead per call. Fine offline. |
| DuckDB `:memory:` | 1,719–1,782 | 75.3 MB | **No** for inner loop — OLAP per-query planning kills point-lookup latency (~500× slower than NumPy). **Yes** for the offline build path (Parquet predicate-pushdown to select the run's subset). |

**Decision per framework:**
- **NumPy columnar → inference inner loop.** Tiny point lookups, sub-µs to ~5 µs, flat scaling, L2/L3-friendly contiguous arrays.
- **DuckDB / PyArrow → offline build/storage only.** Their per-query dispatch overhead (hundreds of µs to ms) makes them the *slowest* options for the inner loop, but they are excellent for Parquet predicate-pushdown when materializing the subset at build time. PyArrow IPC / `.npz` is the natural serialization format for fast cold start.
- **Polars → reject for inner loop.** Whole-frame scan or ~180 µs dispatch per point lookup; fine as an offline batch/corpus tool.
- **SQLite → stays the source of truth and on-disk artifact.** Never touched on the hot path.

So the answer to "keep SQLite for build/storage + in-memory columnar for inference?" is **yes, explicitly two-tier.** The critical refactor is not the storage swap — it is **returning contiguous arrays and never instantiating per-line `Transition` objects** (the ~5.6 ms `iterrows()` materialization, not the engine, is what blocks sub-ms today).

---

## 3. Design — drop-in `AtomicDataSource` inference backend

The codebase already has the right seam. `AtomicDataSource` (`cflibs/core/abc.py`) is a `@runtime_checkable` Protocol with four abstract methods (`get_transitions`, `get_energy_levels`, `get_ionization_potential`, `get_available_elements`), and production code **duck-types** the concrete `AtomicDatabase`. A new `InMemoryColumnarSource` satisfies the Protocol structurally — no inheritance needed — and is constructed once at pipeline entry.

**Critically, the needed species set is known up-front.** `cflibs/inversion/pipeline.py:198` declares `elements` "the run's identity, not overridable" (CLI `--elements Fe Cu`); line-ID narrows it further; the wavelength window is fixed per run. So the entire relevant atomic subset (species × ions × wl-window) can be pre-loaded before any hot loop. The forward and Bayesian paths already exploit this via `snapshot()` / `_AtomicSnapshot.from_solver` (zero SQL in the MCMC loop); the legacy iterative-CPU, Saha, and identify modules still hit SQLite per call — those are the consumers this backend rescues.

```python
class InMemoryColumnarSource:  # duck-types AtomicDataSource — no inheritance
    def __init__(self, db_path, elements, wavelength_range, *, dtype=np.float64):
        # one bulk SELECT ... WHERE element IN (...); group by (element, sp_num);
        # store wavelength-sorted contiguous column arrays per species + scalar dicts
        ...

    @classmethod
    def from_database(cls, db, elements, wavelength_range): ...

    # ---- Protocol surface (zero SQL after construction) ----
    def get_transitions(...) -> list[Transition]: ...      # lazy materialize only if asked
    def get_energy_levels(...) -> list[EnergyLevel]: ...
    def get_ionization_potential(self, el, stage): return self._ip.get((el, stage))
    def get_available_elements(self): ...

    # ---- fast columnar surface (preferred by hot loops, no objects) ----
    def line_columns(self, el, stage, wl_min=None, wl_max=None) -> dict[str, np.ndarray]:
        cols = self._lines[(el, stage)]
        if wl_min is None: return cols
        wl = cols["wavelength_nm"]
        lo, hi = np.searchsorted(wl, [wl_min, wl_max])
        return {c: v[lo:hi] for c, v in cols.items()}      # ~4.7 µs, no base copy
    def partition_coeffs(self, el, stage): return self._pf.get((el, stage))
```

Convenience methods (`partition_function_for`, `snapshot`) are `hasattr`-guarded and not part of the Protocol, so they can be added to keep existing hot loops working unchanged. This generalizes the `AtomicSnapshot` pytree (already used by the forward/Bayesian JAX kernels) to also serve the scalar/legacy consumers. **Best practice:** evaluate `U(T)` from a precomputed `(n_species, 5)` coefficient matrix in one vectorised `polyval` call (**27.7 µs for all 20 species-stages, ~1.4 µs each**) rather than per-species direct-sum — replaces the dominant per-iteration cost entirely.

### In-RAM footprint (measured / projected)

| Scope | Lines | f64 | f32 | Build time |
|---|---:|---:|---:|---:|
| Full reference DB | 28,727 | 2.19 MiB | 1.10 MiB | 117 ms |
| Typical 12-element × ions 1–3, 200–900 nm | 12,271 | **0.96 MiB** | — | **46 ms** (one bulk query) |
| VALD/Kurucz 1M lines | 1,000,000 | 76 MiB | 38 MiB | ~3.2 s one-time |
| VALD/Kurucz 5M lines | 5,000,000 | 381 MiB | 191 MiB | — |
| + 30M molecular TiO (6 f32 cols) | 30,000,000 | — | **0.67 GiB** | — |

Footprint is a non-issue: <1 MiB today, ≤0.67 GiB for the full future atomic+molecular target. Keep `wavelength_nm`/`ek_ev`/`ei_ev` in **f64** (f32's ~7 sig figs ≈ 0.0001 nm error at 500 nm is borderline for line centers) and demote only `aki`/`g`/`rel_int`/Stark columns to f32 if RAM ever matters. `np.memmap`/Arrow-IPC backing is the natural next step if footprint grows — lookup latency unchanged.

---

## 4. Physics-only constraint check

**Clean.** The banned list (`pyproject.toml`, TID251 + the `cflibs/evolution/evaluator.py` AST scanner) is ML-only: `sklearn`, `torch`, `tensorflow`, `keras`, `flax`, `equinox`, `transformers`, `jax.nn`, `jax.experimental.stax`.

- **NumPy** — already a core dependency; the recommended framework. Not banned.
- **PyArrow (≥15), DuckDB (≥0.10)** — already declared deps (offline-tier only). Not banned.
- **Polars** — unlisted but equally non-banned (rejected for the inner loop on latency grounds, not policy).

None embed a server; all are in-process. No new dependency is required for the recommended path (NumPy columnar). All benchmark candidates passed the blocklist check.

---

## 5. Fit with the roadmap

This is the **M10 (latency-last) lever** and explicitly an **investigation/ADR, not an implementation order.** Per the standing directive (*accuracy/precision/reliability first; sub-ms latency deferred until the pipeline is robust*), this representation change must not precede the accuracy work:

- **The DB build comes first.** Real-data accuracy is currently atomic-data-limited (line-list mismatch ~0.171 vs algorithm floor ~0). The M5 Kurucz/VALD ingest (`scripts/ingest_kurucz_atomic.py`, ExoJAX AdbKurucz) and STARK-B `n_e` data are the accuracy levers. A faster lookup over the *wrong* line list buys nothing.
- **This change is accuracy-neutral and forward-compatible.** Because the bottleneck is object materialization (not the engine), and because a VALD/Kurucz DB *explodes the per-row cold-materialization cost roughly linearly* (the 187 ms / 2439-line figure → seconds per heavy species at millions of lines), the columnar pre-materialization becomes *more* urgent exactly when the accuracy-driven DB grows. It is the right representation to land once the line list is settled.

**Concrete next prototype step:** Build `InMemoryColumnarSource.from_database(db, elements, wl_range)` as a standalone module (no shipped-code edits yet), wire it behind a feature flag in **one** consumer — the legacy iterative-CPU `_evaluate_partition_functions` path — using the precomputed `(n_species, 5)` coefficient matrix for vectorised `U(T)`. Benchmark that single seam end-to-end.

**Expected speedup:** `partition_function_for` drops from ~146 µs/species-stage to ~1.4 µs (~100×), removing the **3.82 ms (29%)** partition share of a 13.6 ms solve and a larger share at the 20-iteration cap. Then point the identify modules' wavelength-window lookups at `line_columns()` to convert the **5,591 µs/window** cold-miss identify pattern to **~4.7 µs** (~1,000×). Together these move the binding line-window lookup from 676 µs–28 ms into the **0.14–4.7 µs** range — comfortably inside the sub-ms budget — with the dominant remaining solve cost being the physics math, not data access.

---

## 6. ADR draft

> Drop into `docs/adr/ADR-0007-in-memory-columnar-atomic-source.md`.

```markdown
# ADR-0007: In-Memory Columnar Atomic-Data Source for Real-Time Inference

Status: Proposed (investigation complete; implementation deferred to M10, post accuracy/DB-build)
Date: 2026-06-23
Deciders: CF-LIBS core
Related: ADR-0004 (jittable inversion pipeline), ADR-0006 (instrument calibration
first-class), AtomicSnapshot (core/jax_runtime.py), AtomicDataSource (core/abc.py)

## Context

Real-time CF-LIBS needs sub-ms atomic-data access on the inference hot path.
Profiling the current SQLite + connection-pool + LRU layer (libs_production.db,
28,727 lines, 158 species, CPU) shows the latency is NOT in the database engine:

- Raw `sqlite execute+fetchall` for a line window = 7.2 ms cold; the connection
  pool acquire/release = 4.3 µs (negligible). Warm cached scalar lookups = ~6 µs.
- The cost is per-call Python object materialization that runs even on cache hits:
  (1) `get_transitions()` COLD spends 92% of 187–203 ms in pandas `iterrows()` +
  per-row `Transition` dataclass construction; (2) `partition_function_for().at(T)`
  rebuilds a provider (`to_provider()` 82 µs) and re-runs the exact direct-sum over
  hundreds of levels (60 µs) on every per-iteration call — ~3.82 ms (29%) of a
  13.6 ms 10-element solve, scaling ~5× at the 20-iteration cap.
- The LRU is keyed on exact (element, stage, wl_min, wl_max); the realistic
  identify pattern of distinct per-peak windows runs at 5,591 µs/call, 0.00 hit rate.

The relevant atomic subset (species × ions × wavelength window) is known up-front:
`--elements` is the run's identity and the wavelength window is fixed per run. The
forward and Bayesian paths already pre-load it once via `snapshot()`; the legacy
iterative-CPU, Saha, and identify modules still hit SQLite per call.

Benchmarked storage alternatives (20k windowed lookups, retrieval kernel):
NumPy dict + searchsorted = 3.23 µs (winner, 13.6 MB RSS, flat to 2M lines);
indexed SQLite :memory: = 153 µs (~47×); Polars filter = 184 µs; DuckDB :memory:
= ~1,720 µs (~500×). DuckDB/Polars per-query dispatch makes them the slowest for
tiny point lookups.

## Decision

Adopt a two-tier representation:

1. SQLite (or Parquet) remains the offline BUILD/STORAGE source of truth; the hot
   path never touches it.
2. An `InMemoryColumnarSource` (duck-types the `AtomicDataSource` Protocol) loads
   the run's known-up-front species × ion × wavelength subset ONCE (~46 ms for a
   12-element run) into per-species contiguous NumPy column arrays + scalar dicts
   (IP, partition coeffs, mass), keyed by (element, sp_num). Lookups = dict hash +
   `np.searchsorted` + zero-copy slice. Partition U(T) is evaluated from a
   precomputed (n_species, 5) coefficient matrix via one vectorised polyval call.

The hot path returns contiguous arrays and never instantiates per-line `Transition`
objects. NumPy is the inference tier; DuckDB/PyArrow/Polars are confined to the
offline build path (Parquet predicate-pushdown to select the subset).

## Consequences

Positive:
- Line-window lookup: 676 µs–28 ms → 0.14–4.7 µs (~140× vs raw SQL, ~6,000× vs the
  production pandas/object path). IP 0.13 µs, partition U(T) ~1.4 µs.
- partition_function_for drops ~100× (~146 µs → ~1.4 µs/species-stage), removing the
  29% partition share of the solve.
- Latency flat to millions of lines (hash + binary search); the win grows as the
  accuracy-driven VALD/Kurucz DB grows (cold materialization scales ~linearly with
  per-species line count).
- Footprint trivial: <1 MiB today; ≤0.67 GiB (f32) for the full atomic + 30M
  molecular TiO target. Use f32 only for aki/g/rel_int/Stark; keep wavelength and
  energies in f64 (line-center precision).
- Physics-only clean: NumPy/PyArrow/DuckDB are existing non-banned deps; no ML.
- Minimal blast radius: structural Protocol satisfaction, constructed once at
  pipeline entry, generalizes the existing AtomicSnapshot to the legacy consumers.

Negative / risks:
- Build-time must reuse cached partition_spec_for fits (or precompute PF coeffs at
  build) or the per-species direct-sum FITTING (3.5 s cold snapshot) dominates load.
- `get_transitions(...) -> list[Transition]` object materialization must stay off
  the hot loop; hot loops use the raw `line_columns()` array API.
- f32 wavelength is too coarse for line centers — keep wavelength/energies f64.
- get_stark_parameters (355 µs, unindexed ABS() scan) is a latent forward-Stark cost
  to fold into the columnar source if a per-line Stark path goes hot.

## Sequencing

This is the M10 latency-last lever. Accuracy work (the Kurucz/VALD DB build,
STARK-B n_e) comes FIRST; a faster lookup over the wrong line list buys nothing.
First prototype: standalone `InMemoryColumnarSource.from_database`, flag-gated into
the iterative-CPU `_evaluate_partition_functions` path with vectorised U(T), then
the identify wavelength-window lookups. No shipped-code edits until accuracy lands.
```

---

**Relevant existing seams referenced (absolute paths):**
- `/home/brian/code/CF-LIBS-improved/cflibs/core/abc.py` — `AtomicDataSource` Protocol (4 methods, `@runtime_checkable`)
- `/home/brian/code/CF-LIBS-improved/cflibs/atomic/database.py` — `AtomicDatabase`, `get_transitions`, `partition_function_for`, `snapshot`
- `/home/brian/code/CF-LIBS-improved/cflibs/core/jax_runtime.py` — `AtomicSnapshot` (existing columnar pytree to generalize)
- `/home/brian/code/CF-LIBS-improved/cflibs/inversion/solve/iterative.py` — `_evaluate_partition_functions` (dominant per-iteration consumer), legacy CPU path
- `/home/brian/code/CF-LIBS-improved/cflibs/inversion/pipeline.py:198` — elements "run's identity, not overridable" (justifies up-front pre-load)
- **Next ADR number is ADR-0007** (ADR-0006 is the latest in `docs/adr/`)