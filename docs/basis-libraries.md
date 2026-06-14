# Pre-built basis libraries (T2.4)

Basis-driven identification workflows
(`spectral_nnls`, `hybrid_intersect`, `hybrid_union`,
`nnls_concentration_threshold`) decompose a measured spectrum as a
non-negative linear combination of per-element basis vectors evaluated
at the closest (T, n_e) grid point in a pre-computed library. Each
library is keyed by the instrumental FWHM that was baked in when the
library was generated.

Because each library is ~100-150 MB compressed and the canonical grid
spans 8 FWHM values, we **build the library once and share it over
NFS** rather than rebuilding on every compute node.

## TL;DR — using the canonical libraries

The runner picks up the NFS-shared libraries automatically when no
`--basis-dir` is given. From any vasp node:

```bash
# Confirm the NFS share is mounted (one-time):
ls /cluster/shared/cf-libs-bench/basis_libraries/

# Run a benchmark with basis-driven identifiers:
python scripts/run_unified_benchmark.py \
    --id-workflows spectral_nnls,hybrid_intersect
# --basis-dir is now optional; defaults to the NFS share.
```

If the NFS path is unavailable, override with the environment variable:

```bash
CFLIBS_BASIS_DIR=/some/local/dir python scripts/run_unified_benchmark.py ...
```

## Canonical layout

```
/cluster/shared/cf-libs-bench/basis_libraries/
├── basis_fwhm_0.05nm.h5    # ~120 MB, FWHM = 0.05 nm  (RP ~11k @ 550 nm)
├── basis_fwhm_0.1nm.h5     #
├── basis_fwhm_0.17nm.h5
├── basis_fwhm_0.25nm.h5
├── basis_fwhm_0.5nm.h5
├── basis_fwhm_0.71nm.h5
├── basis_fwhm_1nm.h5
├── basis_fwhm_1.67nm.h5    # ~120 MB, FWHM = 1.67 nm  (RP ~330)
└── manifest.json           # provenance + sha256 of every file
```

Filenames match the loader pattern in
[`cflibs.benchmark.unified.UnifiedBenchmarkContext._basis_files`][unified]
(`basis_fwhm_*nm.h5`). Each file's FWHM-tag is extracted via regex,
so any FWHM-tag whose ``%g`` formatted form is a valid filename works
(`0.05`, `0.1`, `0.17`, etc.).

## Canonical configuration

The builder bakes the following parameters into every library
(`scripts/build_basis_library.py:CanonicalConfig`):

| Parameter | Value | Rationale |
|---|---|---|
| `wavelength_range` | (200.0, 900.0) nm | Covers UV+visible+NIR; standard Aalto LIBS detector range. |
| `pixels` | 4096 | Matches all production benchmark spectra (avoids resample). |
| `temperature_range` | (4000.0, 15000.0) K | Spans calibration cell + plasma-jet conditions. |
| `temperature_steps` | 30 | Nearest-neighbour error <= 1/29 of T-range ~= 380 K. |
| `density_range` | (1e15, 5e17) cm^-3 | Log-spaced; covers 2.5 decades. |
| `density_steps` | 10 | Sufficient for log-bilinear interp in (T, log10(ne)). |
| `ionization_stages` | (1, 2) | Neutrals + singly-ionised; double-ionised is rare in LIBS. |
| FWHM grid | [0.05, 0.10, 0.17, 0.25, 0.50, 0.71, 1.00, 1.67] nm | Mirrors `RP_TO_FWHM` in `scripts/archive/hpc-campaign/run_benchmark_sweep.py`. |

The 30 x 10 = 300-point (T, n_e) grid matches
`scripts/benchmark_element_id.py`'s default and is the minimum required
to keep nearest-neighbour mismatch error below ~5% on simulated steel
spectra. Bump to (50, 20) for tighter interpolation at ~5x build cost.

## Re-building from scratch

```bash
# Default (~100 min on a single CPU): rebuild every FWHM into the
# canonical NFS path.
python scripts/build_basis_library.py

# Force overwrite of existing files:
python scripts/build_basis_library.py --force

# Single FWHM (useful for a focused experiment):
python scripts/build_basis_library.py --fwhm 0.5 --force

# Custom output dir (e.g., a per-experiment slice):
python scripts/build_basis_library.py --output-dir /scratch/cf-libs-bench/basis_libraries_v2

# Just regenerate manifest.json from existing files (cheap, ~seconds):
python scripts/build_basis_library.py --manifest-only
```

### Reproducibility

The builder writes `manifest.json` alongside the HDF5 files containing:

* `built_at_utc` -- UTC timestamp.
* `atomic_db.sha256` -- hash of `libs_production.db` used as input.
* `config` -- the full `CanonicalConfig` dict.
* `files[].sha256` -- hash of every generated file.

A second `--manifest-only` run with the same inputs produces an
identical `manifest.json` modulo `built_at_utc`. Per-file sha256
values are deterministic *because* the underlying Saha-Boltzmann
solver is deterministic; if a re-build produces different hashes,
either the atomic database changed or a solver code path was
edited -- both should trigger a CI alarm.

## How `run_unified_benchmark.py` finds the libraries

Resolution order (highest priority first):

1. Explicit `--basis-dir <path>` on the CLI.
2. `$CFLIBS_BASIS_DIR` environment variable.
3. `/cluster/shared/cf-libs-bench/basis_libraries` -- iff the NFS
   parent directory exists.
4. `output/basis_libraries` -- in-repo legacy default.

The resolver lives in `scripts/run_unified_benchmark._resolve_basis_dir`.
The builder script (`scripts/build_basis_library.resolve_basis_dir`)
uses the same priority order so a typical workflow is::

    # On the build host
    python scripts/build_basis_library.py        # writes to NFS share
    # On a compute host (e.g. vasp-01)
    python scripts/run_unified_benchmark.py ...  # reads from NFS share

## When to rebuild

You need to regenerate basis libraries when **any** of these change:

* The atomic database (`libs_production.db`) -- new transitions, new
  ionization levels, new partition functions.
* `CanonicalConfig` values in `scripts/build_basis_library.py` --
  wavelength range, T/n_e grid, pixels, etc.
* Code paths inside `cflibs.manifold.basis_library` or
  `cflibs.plasma.saha_boltzmann` -- the per-element spectra are
  computed from first principles, so a solver bug-fix invalidates
  every previously cached library.

In all three cases:

```bash
python scripts/build_basis_library.py --force
```

Then commit the **new** `manifest.json` -- it serves as the canonical
record of which library version is on the NFS share.

[unified]: ../cflibs/benchmark/unified.py
