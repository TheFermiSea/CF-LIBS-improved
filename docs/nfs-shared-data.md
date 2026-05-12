# NFS-Shared Benchmark Data (T1.3)

> Status: **active** since 2026-05-12
> Issue: `CF-LIBS-improved-mg58` (T1.3) — parent epic `CF-LIBS-improved-7lht`
> Branch: `feat/nfs-shared-data`

## Summary

The per-node `/scratch/cf-libs-bench/repo/data/` directories (one copy per vasp
node, ~11 GB each, 33 GB total) are replaced by a single NFS export at
`/cluster/shared/cf-libs-bench/data/` exported from `slurm-ctl` (10.0.0.5) and
mounted read-write on all three vasp nodes.

## Layout

```
slurm-ctl:/cluster/shared/cf-libs-bench/data/
├── aalto_libs/
├── bhvo2_usgs/
├── chemcam_calib/
├── csa_planetary_libs/
├── gibbons2024_nitrogen_libs/
├── nist_srm_612/
├── nist_steel/
├── silva2022_tropical_soils/
└── vrabel2020_soil_benchmark/
    ├── train.h5            (~7.5 GB)
    ├── test.h5             (~3.2 GB)
    └── ...
```

Total: ~11 GB at the time of writing. The NFS export `/cluster/shared` is
already mounted on every vasp node (and on `ai-proxy` for indirect access via
slurm-ctl), so no client-side mount changes are required.

## Rationale

| Before T1.3 | After T1.3 |
|-------------|------------|
| 3 × 11 GB = 33 GB duplicated across vasp-01/02/03 `/scratch` | 1 × 11 GB on slurm-ctl, mounted everywhere |
| New data must be `rsync`'d to every node | Drop a file once on slurm-ctl, all nodes see it |
| Per-node drift if rsync skipped or partial | Single source of truth; md5 verified identical |
| `data/vrabel2020_soil_benchmark/train.h5` lookup hits local disk (fast) | Same lookup hits NFS over 10G Ethernet (~110 MB/s) |

For the 6-cell parallel benchmark, the I/O bottleneck has historically been
parsing the H5 files (Python-side, CPU-bound), not the read itself, so the
move to NFS is expected to be cost-neutral on hot-cached data and to save
~22 GB of disk after old `/scratch/cf-libs-bench/repo/data/` copies are
cleaned up.

## How code finds the data

The unified benchmark and its driver scripts now all honor the same
`CFLIBS_DATA_DIR` environment variable:

| Component | Default | Override |
|-----------|---------|----------|
| `scripts/run_unified_benchmark.py` | `$CFLIBS_DATA_DIR` if set, else `data` | `--data-dir <path>` CLI flag |
| `scripts/run_cell.sh` | `/cluster/shared/cf-libs-bench/data` | `CFLIBS_DATA_DIR=/path/to/data ./run_cell.sh …` |
| `scripts/loop_24h_driver.sh` | `/cluster/shared/cf-libs-bench/data` | `CFLIBS_DATA_DIR=/path/to/data ./loop_24h_driver.sh …` |
| `scripts/loop_iteration.sh` | `/cluster/shared/cf-libs-bench/data` (on the remote) | `BENCH_DATA_DIR=/path/to/data ./loop_iteration.sh …` |

All three shell scripts fall back to the repo-relative `data/` directory if
the canonical NFS path is missing — preserving offline / laptop workflows.

## Initial population

```bash
# Create the directory on the NFS server.
ssh root@10.0.0.5 'mkdir -p /cluster/shared/cf-libs-bench/data && chmod 2775 /cluster/shared/cf-libs-bench /cluster/shared/cf-libs-bench/data'

# Populate from the canonical vasp-03 source.
ssh root@10.0.0.22 'rsync -av /scratch/cf-libs-bench/repo/data/ root@10.0.0.5:/cluster/shared/cf-libs-bench/data/'
```

This was executed on 2026-05-12. Total transfer: 10,879,242,424 bytes
(~10.1 GiB) at ~110 MB/s over 10 GbE.

## Verification

```bash
# 1. File visible from every vasp node.
for h in 10.0.0.20 10.0.0.21 10.0.0.22; do
  ssh root@$h 'ls /cluster/shared/cf-libs-bench/data/vrabel2020_soil_benchmark/train.h5'
done

# 2. md5 matches across nodes.
for h in 10.0.0.20 10.0.0.21 10.0.0.22; do
  ssh root@$h 'md5sum /cluster/shared/cf-libs-bench/data/vrabel2020_soil_benchmark/train.h5'
done
# All three return: dbf195ed7d4455a21de2e40bdfea849b
```

## Cleanup of per-node /scratch copies

**Do not delete the per-node /scratch/cf-libs-bench/repo/data directories until
the in-flight 6-cell experiment has finished and at least one fresh benchmark
run has been validated against the NFS path.**

Once validated, the per-node copy can be replaced with a symlink:

```bash
for h in 10.0.0.20 10.0.0.21 10.0.0.22; do
  ssh root@$h '
    cd /scratch/cf-libs-bench/repo &&
    mv data data.deprecated-$(date +%Y%m%d) &&
    ln -s /cluster/shared/cf-libs-bench/data data
  '
done

# After ~1 week of clean runs, remove the deprecated copy:
for h in 10.0.0.20 10.0.0.21 10.0.0.22; do
  ssh root@$h 'rm -rf /scratch/cf-libs-bench/repo/data.deprecated-*'
done
```

Reclaims ~33 GB total across the cluster.

## Rollback

If NFS performance proves to be a bottleneck (it shouldn't be for the H5
parsing workload), revert by simply un-setting `CFLIBS_DATA_DIR` and ensuring
each node has a local `/scratch/cf-libs-bench/repo/data/` copy — the scripts'
fallback logic handles the rest. The NFS export can remain in place at
negligible cost.

## See also

- `tests/scripts/test_nfs_data_dir.py` — smoke test that the canonical path
  exists and is non-empty when invoked from a vasp node.
- `bd show CF-LIBS-improved-mg58` — issue tracking this work.
- `bd show CF-LIBS-improved-7lht` — parent epic (cluster I/O modernization).
