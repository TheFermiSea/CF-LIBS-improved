# Dataset-sharding design consultation (T2.2)

Synthesis of Codex (`gpt-5.3-codex`) + Gemini (`gemini-3-flash-preview`)
consultation on the `--dataset-shard N/K` design for splitting the
Vrabel 2020 50,000-spectrum benchmark across the 3-node cluster.

## Question 1 — Stride (`spectra[N-1::K]`) vs contiguous (`spectra[(N-1)*chunk:N*chunk]`)

**Codex:**

* Determinism on re-run: both are deterministic if input ordering is fixed
  by the loader. For invariance across different `K`, would need a
  hash-bucket scheme (`hash(spectrum_id) % K`), but that's overkill for our
  use case where shard configurations are fixed per experiment.
* Load balance with unequal shot counts: **stride wins** — statistically
  mixes heavy/light samples across shards; contiguous blocks can
  concentrate "heavy" regions on one node.
* Cache / I/O locality: **contiguous wins** — sequential HDF5 reads,
  better upstream cache. Stride scatters reads.
* Practical recommendation: stride if per-spectrum compute dominates;
  contiguous if I/O/preprocessing dominates.

**Decision:** Use **stride** (`spectra[N-1::K]`).

* The Vrabel HDF5 read happens *once* in the loader on each node anyway
  (we materialise the full BenchmarkDataset in memory then slice), so
  the I/O-locality argument doesn't apply.
* Per-spectrum compute dominates downstream — JAX kernel inversion takes
  ~1-2 s per spectrum vs. ~0.1 ms to bring it from a numpy array into
  the runner.
* Sample-balance matters: the Vrabel benchmark has 100 samples, but only
  some samples carry the harder mineralogies. Stride hits each sample
  proportionally on every shard.
* The stride is also the most-intuitive and easiest-to-audit partition:
  shard 1/3 gets indices `[0,3,6,9,...]`, shard 2/3 gets `[1,4,7,...]`,
  shard 3/3 gets `[2,5,8,...]`. Disjoint cover trivially verifiable.

## Question 2 — Shard before or after per-sample shot capping

**Codex:** "Apply `max_spectra_per_sample` **before** sharding (globally).
This keeps per-sample limits consistent across shards and avoids
shard-dependent composition."

**Decision:** Apply per-sample capping inside the loader **first**, then
shard. The semantics that matter:

* `--vrabel-max-shots 10 --dataset-shard 1/3`: load 10 shots × 100
  samples = 1000 spectra, then shard 1/3 ≈ 333. Small smoke run still
  divides cleanly.
* `--vrabel-max-shots 0 --dataset-shard 1/3`: load all 50,000, then
  shard 1/3 ≈ 16,667 — the headline use case.

This matches the spec: "shard AFTER per-sample shot capping (so
`--vrabel-max-shots 0` + `--dataset-shard 1/3` means full corpus / 3,
not capped / 3)." Capping at *load time* and sharding at the BenchmarkDataset
boundary cleanly separates concerns: the loader doesn't know about
shards (and won't need to), and the shard plumbing doesn't need to
understand per-sample structure.

## Question 3 — Cross-shard seeding

**Codex:** "Don't use only `seed_base + N` if you want invariance to
shard count/layout. Use a global seed and derive RNG per item from
stable IDs/index. `seed_base + N` is fine only when you need
independent shard streams, not identical global outcomes across
different sharding setups."

**Decision:** Use **the same seed** for every shard. Justification:

* Our shard partition is deterministic given the seed-independent ordering
  of `BenchmarkDataset.spectra` (sorted by `sample_id` in the loader),
  so seed isn't load-bearing for the partition itself.
* Per-spectrum inversion (JAX kernels) is deterministic — the seed only
  matters for the small RNG-using paths (outlier injection in
  `--perturb`, bootstrap CI in summarisation). Those should be
  **identical** across shards so that aggregator-side bootstrap stays
  internally consistent. Using the same seed accomplishes that.
* Acceptance test enforced: running 1/3 + 2/3 + 3/3 with seed=42
  produces 3 disjoint sets that union to the full corpus. Verified in
  `tests/benchmark/test_dataset_shard.py::test_shard_partition_is_disjoint_cover`.

## Question 4 — Aggregator design (Gemini)

* **Add `shard_id` column** (1, 2, 3) on merge — critical for data
  lineage, debugging worker-specific hardware biases, verifying
  partition correctness.
* **Fail-fast on overlap**: if two shards write the same
  `(spectrum_id, id_workflow_name, composition_workflow_name, outer_split_id)`
  tuple, aggregator errors out. Silent dedupe masks partitioner bugs.
* **Bootstrap-CI math** for pooled summary statistics:
  `SE_pooled = sqrt(sum(n_i * (s_i^2 + (m_i - M)^2)) / N)`
  This is the Law of Total Variance. Correct for linear estimators
  (mean), suspect for non-linear ones (median, percentile). For the
  T3.x downstream summarisation we **re-bootstrap the merged rows**
  rather than pool — once you have the unified Parquet, it's a single
  table and you can bootstrap the merged sample directly.
* **Schema conformance**: validate every shard's Parquet has matching
  schema before concatenation. Mismatches indicate version skew
  between nodes.

## Implementation summary

```python
# loaders.py — capped at load time
shots = arr[:, :n_load]   # per-sample shot cap

# Then in the loader entrypoint (after BenchmarkDataset constructed):
def _apply_shard(dataset, shard_n, shard_k):
    if shard_k == 1:
        return dataset
    sharded_spectra = dataset.spectra[shard_n - 1 :: shard_k]
    # Annotate every spectrum so downstream aggregation can read it back.
    for s in sharded_spectra:
        s.annotations = {**s.annotations, "shard_n": shard_n, "shard_k": shard_k}
    return BenchmarkDataset(... spectra=sharded_spectra ...)
```

```python
# aggregate_shards.py — concat + shard_id column + overlap check
table = pa.concat_tables(per_shard_tables)
duplicates = (
    table.to_pandas()
        .groupby(["spectrum_id", "id_workflow_name", "composition_workflow_name",
                  "outer_split_id"])
        .size()
)
if (duplicates > 1).any():
    raise ValueError("Shard overlap detected; partitioner bug.")
```
