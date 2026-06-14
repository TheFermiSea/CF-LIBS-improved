#!/usr/bin/env python3
"""Build canonical basis libraries for spectral_nnls / hybrid / NNLS-threshold workflows.

This is the *canonical* single-source-of-truth builder. It produces files
named ``basis_fwhm_<X.X>nm.h5`` matching the loader pattern in
:class:`cflibs.benchmark.unified.UnifiedBenchmarkContext` (and the related
sweep scripts under ``scripts/archive/hpc-campaign/``).

Output layout (canonical NFS path - mounted on every vasp node)::

    /cluster/shared/cf-libs-bench/basis_libraries/
    ├── basis_fwhm_0.05nm.h5
    ├── basis_fwhm_0.1nm.h5
    ├── basis_fwhm_0.17nm.h5
    ├── basis_fwhm_0.25nm.h5
    ├── basis_fwhm_0.5nm.h5
    ├── basis_fwhm_0.71nm.h5
    ├── basis_fwhm_1.0nm.h5
    ├── basis_fwhm_1.67nm.h5
    └── manifest.json     # provenance: db hash, config, build timestamp

Re-running the builder is a no-op when every expected file already exists
on disk (the legacy ``--force`` flag rebuilds in-place).

Default knobs match the canonical Aalto-grade configuration:

* wavelength_range  (200.0, 900.0) nm, 4096 pixels
* temperature_range (4000.0, 15000.0) K, 30 steps
* density_range     (1e15, 5e17) cm^-3, 10 log-spaced steps
* ionization_stages (1, 2)
* FWHM grid         [0.05, 0.10, 0.17, 0.25, 0.50, 0.71, 1.00, 1.67] nm
                    (mirrors ``RP_TO_FWHM`` in ``scripts/archive/hpc-campaign/run_benchmark_sweep.py``)

The 30 x 10 = 300-point (T, n_e) grid is the same one
``scripts/benchmark_element_id.py`` uses by default. It is plenty for
nearest-neighbour / bilinear lookup in the basis selector and keeps each
HDF5 file under ~150 MB compressed (~1.3 GB total across 8 FWHMs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Canonical FWHM grid (nm). Order matters only for log clarity; the loader
# is order-insensitive. Stays in sync with run_benchmark_sweep.RP_TO_FWHM.
CANONICAL_FWHM_GRID: List[float] = [0.05, 0.10, 0.17, 0.25, 0.50, 0.71, 1.00, 1.67]

# Canonical NFS share - mounted at /cluster/shared on every vasp node.
CANONICAL_NFS_DIR = Path("/cluster/shared/cf-libs-bench/basis_libraries")


@dataclass(frozen=True)
class CanonicalConfig:
    """Frozen build parameters. Changing any of these invalidates the cache."""

    wavelength_min_nm: float = 200.0
    wavelength_max_nm: float = 900.0
    pixels: int = 4096
    temperature_min_K: float = 4000.0
    temperature_max_K: float = 15000.0
    temperature_steps: int = 30
    density_min_cm3: float = 1e15
    density_max_cm3: float = 5e17
    density_steps: int = 10
    ionization_stages: tuple = (1, 2)

    def as_dict(self) -> dict:
        return {
            "wavelength_range_nm": [self.wavelength_min_nm, self.wavelength_max_nm],
            "pixels": self.pixels,
            "temperature_range_K": [self.temperature_min_K, self.temperature_max_K],
            "temperature_steps": self.temperature_steps,
            "density_range_cm3": [self.density_min_cm3, self.density_max_cm3],
            "density_steps": self.density_steps,
            "ionization_stages": list(self.ionization_stages),
        }


def basis_filename(fwhm_nm: float) -> str:
    """Canonical basis-library filename for a given FWHM (nm).

    The format ``basis_fwhm_<g>nm.h5`` matches the glob/regex used by
    :class:`cflibs.benchmark.unified.UnifiedBenchmarkContext._basis_files`::

        glob: basis_fwhm_*nm.h5
        rgx:  basis_fwhm_([0-9.]+)nm\\.h5$

    ``%g`` formatting drops trailing zeros (0.10 -> 0.1, 1.00 -> 1.0) which
    is also what the back-compat fallback in
    ``scripts/archive/hpc-campaign/run_benchmark_sweep._find_basis_path`` searches for.
    """
    return f"basis_fwhm_{fwhm_nm:g}nm.h5"


def _resolve_db_path(arg: str | None) -> Path:
    if arg is not None:
        path = Path(arg).expanduser().resolve()
        if not path.exists():
            sys.exit(f"ERROR: atomic database not found: {path}")
        return path
    candidates = [
        PROJECT_ROOT / "ASD_da" / "libs_production.db",
        PROJECT_ROOT / "libs_production.db",
        Path.cwd() / "ASD_da" / "libs_production.db",
        Path.cwd() / "libs_production.db",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    sys.exit(
        "ERROR: atomic database not found. Pass --db-path or place "
        "libs_production.db under ASD_da/."
    )


def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _build_one(
    db_path: Path,
    output_dir: Path,
    fwhm_nm: float,
    cfg: CanonicalConfig,
    force: bool,
) -> Path:
    from cflibs.manifold.basis_library import BasisLibraryConfig, BasisLibraryGenerator

    out_path = output_dir / basis_filename(fwhm_nm)
    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} already exists ({out_path.stat().st_size / 1e6:.0f} MB)")
        return out_path

    output_dir.mkdir(parents=True, exist_ok=True)

    bl_cfg = BasisLibraryConfig(
        db_path=str(db_path),
        output_path=str(out_path),
        wavelength_range=(cfg.wavelength_min_nm, cfg.wavelength_max_nm),
        pixels=cfg.pixels,
        temperature_range=(cfg.temperature_min_K, cfg.temperature_max_K),
        temperature_steps=cfg.temperature_steps,
        density_range=(cfg.density_min_cm3, cfg.density_max_cm3),
        density_steps=cfg.density_steps,
        ionization_stages=tuple(cfg.ionization_stages),
        instrument_fwhm_nm=fwhm_nm,
    )

    t0 = time.monotonic()
    gen = BasisLibraryGenerator(bl_cfg)
    gen.generate(
        progress_callback=lambda done, total: print(
            f"    [{done}/{total}] elements", end="\r", flush=True
        )
    )
    print()
    dt = time.monotonic() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"  [done] {out_path.name} ({size_mb:.0f} MB, {dt:.1f}s)")
    return out_path


def build_all(
    db_path: Path,
    output_dir: Path,
    fwhm_values: Sequence[float],
    cfg: CanonicalConfig,
    force: bool,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for i, fwhm in enumerate(fwhm_values, start=1):
        print(f"[{i}/{len(fwhm_values)}] FWHM = {fwhm:g} nm")
        written.append(_build_one(db_path, output_dir, fwhm, cfg, force))
    _write_manifest(output_dir, db_path, fwhm_values, cfg)
    return written


def _write_manifest(
    output_dir: Path,
    db_path: Path,
    fwhm_values: Sequence[float],
    cfg: CanonicalConfig,
) -> Path:
    manifest = {
        "schema_version": 1,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "scripts/build_basis_library.py",
        "fwhm_grid_nm": [float(f) for f in fwhm_values],
        "config": cfg.as_dict(),
        "atomic_db": {
            "path": str(db_path),
            "sha256": _hash_file(db_path),
            "size_bytes": db_path.stat().st_size,
        },
        "files": [],
    }
    for fwhm in fwhm_values:
        f = output_dir / basis_filename(fwhm)
        if f.exists():
            manifest["files"].append(
                {
                    "fwhm_nm": float(fwhm),
                    "filename": f.name,
                    "size_bytes": f.stat().st_size,
                    "sha256": _hash_file(f),
                }
            )
    out = output_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {out}")
    return out


def resolve_basis_dir(cli_value: Path | None) -> Path:
    """Resolve --output-dir / CFLIBS_BASIS_DIR / canonical NFS path / repo default.

    Priority (highest first):
      1. Explicit --output-dir on the CLI
      2. CFLIBS_BASIS_DIR environment variable
      3. ``/cluster/shared/cf-libs-bench/basis_libraries`` if mounted
      4. ``<repo>/output/basis_libraries``
    """
    if cli_value is not None:
        return cli_value.expanduser().resolve()
    env = os.environ.get("CFLIBS_BASIS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    if CANONICAL_NFS_DIR.parent.exists():
        return CANONICAL_NFS_DIR
    return (PROJECT_ROOT / "output" / "basis_libraries").resolve()


def _normalize_fwhm(values: Iterable[float] | None) -> List[float]:
    if not values:
        return list(CANONICAL_FWHM_GRID)
    return sorted({float(v) for v in values})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build canonical basis libraries for spectral_nnls / hybrid / "
            "nnls_concentration_threshold identification workflows. "
            "Default output is the canonical NFS share at "
            "/cluster/shared/cf-libs-bench/basis_libraries when mounted, "
            "falling back to <repo>/output/basis_libraries."
        ),
    )
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. If unset, uses $CFLIBS_BASIS_DIR, then "
            "/cluster/shared/cf-libs-bench/basis_libraries when mounted, "
            "then <repo>/output/basis_libraries."
        ),
    )
    parser.add_argument(
        "--fwhm",
        type=float,
        nargs="+",
        default=None,
        help="FWHM values (nm). Default is the canonical 8-point grid.",
    )
    parser.add_argument(
        "--T-steps", type=int, default=CanonicalConfig.temperature_steps
    )
    parser.add_argument(
        "--ne-steps", type=int, default=CanonicalConfig.density_steps
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-build even if the output file already exists.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Skip generation; just (re)write manifest.json from existing files.",
    )
    args = parser.parse_args(argv)

    db_path = _resolve_db_path(args.db_path)
    output_dir = resolve_basis_dir(args.output_dir)
    fwhm_values = _normalize_fwhm(args.fwhm)

    cfg = CanonicalConfig(
        temperature_steps=args.T_steps,
        density_steps=args.ne_steps,
    )

    print(f"Atomic database: {db_path}")
    print(f"Output directory: {output_dir}")
    print(f"FWHM grid (nm):   {fwhm_values}")
    print(f"Config: {cfg.as_dict()}\n")

    if args.manifest_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_manifest(output_dir, db_path, fwhm_values, cfg)
        return 0

    build_all(db_path, output_dir, fwhm_values, cfg, args.force)
    print("\nDone. Basis libraries ready at:")
    print(f"  {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
