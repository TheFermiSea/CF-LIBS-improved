# Dataset Sharding for Parallel Benchmark Execution (T2.2)

`scripts/run_unified_benchmark.py --dataset-shard N/K` splits the Vrabel
2020 50,000-spectrum soil benchmark across multiple cluster nodes so
that each node processes a disjoint slice of the full corpus. After all
K shards complete, `scripts/aggregate_shards.py` merges their
per-shard `results.parquet` files into a single unified table.

## When to use sharding

Use `--dataset-shard` when you want **population-level precision** on
the full Vrabel 50k benchmark but a single node can't process all
50,000 spectra inside the wall-clock window:

* **Before sharding (current cluster setup):** vasp-01 / vasp-02 /
  vasp-03 each run `--vrabel-max-shots 10` (10 shots/sample × 100
  samples = 1,000 spectra/cell). Three identifier×composition cells
  produce 3 × 1,000 = 3,000 spectra of evidence.
* **After sharding:** vasp-01 / vasp-02 / vasp-03 each run
  `--vrabel-max-shots 0 --dataset-shard {1,2,3}/3`. Each cell sees
  ~16,667 spectra of the **same** full corpus. Three shards merge into
  a single 50,000-spectrum result.

The bootstrap-CI math says SE shrinks as `1/sqrt(n)`. Going from 1,000
to 50,000 spectra gives a **7×** CI tightening on every population
statistic (mean d_A, F1, RMSE) for the same wall time.

Do **NOT** use sharding for BHVO-2 (12 spectra) or NIST SRM 612
(<100 spectra) — those are too small to benefit, and per-shard sample
counts would be statistically uninformative. The loader explicitly
skips sharding for these datasets.

## CLI usage

```bash
# Single-shard runs (one per node — different machines)
python scripts/run_unified_benchmark.py \
    --vrabel-max-shots 0 --dataset-shard 1/3 \
    --id-workflows correlation --composition-workflows iterative_jax \
    --output-dir /scratch/cf-libs-bench/shard-1

python scripts/run_unified_benchmark.py \
    --vrabel-max-shots 0 --dataset-shard 2/3 \
    --id-workflows correlation --composition-workflows iterative_jax \
    --output-dir /scratch/cf-libs-bench/shard-2

python scripts/run_unified_benchmark.py \
    --vrabel-max-shots 0 --dataset-shard 3/3 \
    --id-workflows correlation --composition-workflows iterative_jax \
    --output-dir /scratch/cf-libs-bench/shard-3

# Merge into one result
python scripts/aggregate_shards.py \
    /scratch/cf-libs-bench/shard-1/results.parquet \
    /scratch/cf-libs-bench/shard-2/results.parquet \
    /scratch/cf-libs-bench/shard-3/results.parquet \
    --output /scratch/cf-libs-bench/merged/results.parquet
```

`--dataset-shard` accepts `N/K` strings with `1 ≤ N ≤ K ≤ 16`.
`1/1` (the default) is equivalent to "no sharding".

## Partition scheme: stride

Each shard takes a **strided** slice of the loaded
BenchmarkDataset's spectra list:

```text
shard 1/3 → spectra[0::3]   →  indices [0, 3, 6, 9, ...]
shard 2/3 → spectra[1::3]   →  indices [1, 4, 7, 10, ...]
shard 3/3 → spectra[2::3]   →  indices [2, 5, 8, 11, ...]
```

Stride is preferred over contiguous blocks (`spectra[0:n], [n:2n],
[2n:]`) because:

* **Load balance:** if per-sample shot counts are uneven (e.g.
  `--vrabel-max-shots 50` caps the bigger samples), stride mixes heavy
  and light samples evenly across nodes. Contiguous blocks can
  concentrate all heavy samples on one shard.
* **Sample coverage:** every shard sees a proportional sample from each
  of the 100 Vrabel CRMs, so per-CRM statistics are well-defined inside
  every shard.
* **Audit simplicity:** the disjoint cover property
  (`union = full corpus, no overlap`) is trivial to verify from the
  stride formula. The aggregator enforces it as a runtime invariant.

See `docs/archive/dataset-sharding-consultation.md` for the full design
discussion (Codex + Gemini consultation).

## Sharding is applied **after** the per-sample shot cap

The loader caps shots per Vrabel sample first (`--vrabel-max-shots`),
then `apply_dataset_shard` slices the resulting BenchmarkDataset.

This means:

| `--vrabel-max-shots` | `--dataset-shard` | Spectra per shard |
|----------------------|-------------------|-------------------|
| `10`                 | `1/1`             | 1,000 (10×100)    |
| `10`                 | `1/3`             | ~333              |
| `0` (no cap)         | `1/1`             | 50,000            |
| `0` (no cap)         | `1/3`             | ~16,667           |
| `50` (default)       | `1/3`             | ~1,667            |

The headline use case is the **bottom-right cell** — full corpus / 3,
yielding the 7× CI tightening described above.

## Cross-shard determinism

Sharding is **seed-independent**: the partition is fully determined by
`(shard_n, shard_k)` and the deterministic ordering of
`BenchmarkDataset.spectra` (sorted by sample ID inside the Vrabel
loader). Run shard 1/3 today, run shard 1/3 next week, you get the
exact same 16,667 spectra.

Each shard run should pass `--seed <S>` (the same `S` for all K shards
in the same experiment) so that downstream RNG-using paths
(`--perturb`, bootstrap CI in summaries) produce comparable results.
The seed does NOT affect partition membership.

## Per-row provenance

Every shard run writes its `shard_n` / `shard_k` into the parquet's
run-metadata columns (added in v1 schema, see
`docs/results-parquet-schema.md`). Each spectrum also carries
`shard_n` / `shard_k` in `annotations`, surfaced as a JSON-encoded
`annotations_json` column. Downstream consumers can therefore:

```sql
SELECT shard_n, COUNT(*) AS n_spectra, AVG(d_a) AS mean_d_a
FROM read_parquet('merged/results.parquet')
WHERE record_kind = 'composition' AND scored
GROUP BY shard_n
ORDER BY shard_n;
```

…to verify the partition is statistically balanced post-merge.

## Aggregator behaviour

`scripts/aggregate_shards.py`:

1. Loads each input Parquet.
2. Validates that schemas match (no silent column drops).
3. Concatenates with `pyarrow.concat_tables`.
4. **Enforces the disjoint-cover invariant** — if any two shards
   produce a row with the same `(record_kind, spectrum_id,
   id_workflow_name, composition_workflow_name, outer_split_id)` tuple,
   the aggregator exits with code 2 ("Shard overlap detected").
5. Writes the merged Parquet to `--output`.

Exit codes: `0` success, `1` schema mismatch or I/O error, `2`
overlap detected. Pass `--allow-overlap` to bypass the check (only
useful when intentionally merging non-shard files).

## Bootstrap-CI math

Two valid ways to compute population-level confidence intervals on a
sharded run:

### Option A: Pooled summary statistics (Law of Total Variance)

If each shard reports mean `m_i` with standard error `s_i` over `n_i`
spectra, the pooled estimator over `N = Σ n_i` total spectra has:

```
M_pooled = Σ (n_i * m_i) / N
Var_pooled = Σ [n_i * (s_i² + (m_i - M_pooled)²)] / N
```

The cross-shard term `(m_i - M_pooled)²` captures between-shard
variance and is necessary when shard sample counts are unequal.

**Use this when:** all shards report only summary statistics (e.g. for
fast dashboards) and the statistic is a sample mean.

### Option B: Re-bootstrap from the merged Parquet

Once `aggregate_shards.py` produces the merged Parquet, bootstrap the
**merged row table** directly. This is the recommended path:

```python
import duckdb
import numpy as np

con = duckdb.connect()
rows = con.execute("""
    SELECT d_a FROM read_parquet('merged/results.parquet')
    WHERE record_kind = 'composition' AND scored
""").fetchnumpy()["d_a"]

rng = np.random.default_rng(42)
boot_means = np.array([
    np.mean(rng.choice(rows, size=len(rows), replace=True))
    for _ in range(10_000)
])
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
```

**Use this when:** the statistic is non-linear (median, percentile,
rank-correlation, Friedman/Nemenyi ranks). The merged row table is the
canonical input; bootstrap from it.

## Implementation notes

* Loader entry point: `cflibs.benchmark.loaders.apply_dataset_shard`.
* Shard is plumbed through `cflibs.benchmark.unified.load_default_datasets`
  via a new `dataset_shard: tuple[int, int]` kwarg.
* `run_unified_benchmark.py` and `parameter_sweep.py` both accept
  `--dataset-shard N/K`; the latter forwards via `--config-args`.
* Parquet schema v1 gained two new nullable columns: `shard_n`
  (int32), `shard_k` (int32). Existing readers that ignore unknown
  columns are unaffected.
* See `tests/benchmark/test_dataset_shard.py` for the disjoint-cover
  invariant, CLI parser, and aggregator behaviour tests (12 tests,
  <1 s).
