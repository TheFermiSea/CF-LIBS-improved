#!/usr/bin/env python3
"""Export real benchmark spectra + certified compositions to uniform .npz.

Drives the *same* cflibs benchmark dataset adapters the scoreboard uses
(cflibs.benchmark.scoreboard_registry + adapters_core/adapters_extended) and
writes one .npz per dataset that the inversion harness can consume directly.

For each dataset we export, per spectrum:
  - wavelength grid (nm)
  - intensity
  - certified composition (element wt%, the adapter's SpectrumTruth.composition_wt)

.npz layout (per dataset)
-------------------------
  dataset            : str  (scalar)
  spectrum_ids       : (N,) <U...   unique id per spectrum
  uniform_grid       : bool scalar  whether every spectrum shares one grid
  # if uniform_grid:
  wavelength_nm      : (G,) float64        shared grid
  intensity          : (N, G) float64      stacked spectra
  # else (ragged grids):
  wavelength_nm      : (N,) object  each a (Gi,) float64 array
  intensity          : (N,) object  each a (Gi,) float64 array
  grid_sizes         : (N,) int64   Gi per spectrum
  # composition truth (ragged element sets -> dense matrix over the union):
  elements           : (E,) <U...   union of all certified elements (sorted)
  composition_wt     : (N, E) float64   element wt%; NaN where the element is
                                        not certified for that spectrum
  composition_basis  : (N,) <U...   "element_wt" / "presence_only"
  resolving_power    : (N,) float64 per-spectrum RP hint (NaN if unknown)
  notes              : (N,) object  per-spectrum provenance notes

Presence-only datasets (no composition_wt) are still exported: their
composition_wt matrix is all-NaN and elements come from elements_present.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Drive the canonical adapters. Build the registry the scoreboard way.
from cflibs.benchmark.scoreboard_registry import (
    ensure_default_datasets,
    iter_datasets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("export_truth")

# Datasets requested, in priority order (supercam_labcal first).
TARGET_DATASETS = [
    "supercam_labcal",
    "supercam_scct",
    "chemcam_calib",
    "bhvo2_chemcam",
    "csa_planetary",
]


def _grids_uniform(wls: list[np.ndarray]) -> bool:
    if not wls:
        return True
    g0 = wls[0]
    for g in wls[1:]:
        if g.shape != g0.shape or not np.array_equal(g, g0):
            return False
    return True


def export_dataset(name: str, out_dir: Path) -> dict:
    """Run one adapter, build arrays, write <name>.npz. Returns a manifest dict."""
    t0 = time.time()
    entry = next(iter_datasets(names=[name]))
    factory = entry.adapter_factory

    spectrum_ids: list[str] = []
    wls: list[np.ndarray] = []
    intens: list[np.ndarray] = []
    comp_maps: list[dict] = []
    bases: list[str] = []
    rps: list[float] = []
    notes: list[str] = []
    elem_union: set[str] = set()

    n = 0
    for spec_id, wl, inten, truth in factory():
        wl = np.asarray(wl, dtype=np.float64)
        inten = np.asarray(inten, dtype=np.float64)
        spectrum_ids.append(str(spec_id))
        wls.append(wl)
        intens.append(inten)
        comp = dict(truth.composition_wt) if truth.composition_wt is not None else {}
        comp_maps.append(comp)
        bases.append(truth.composition_basis)
        rps.append(float(truth.resolving_power) if truth.resolving_power is not None else np.nan)
        notes.append(truth.notes or "")
        # Union of certified-composition elements AND presence-only elements.
        elem_union.update(comp.keys())
        elem_union.update(truth.elements_present)
        n += 1
        if n % 200 == 0:
            log.info("  %s: %d spectra so far ...", name, n)

    if n == 0:
        log.warning("%s: adapter yielded NOTHING (missing data / skip-with-log).", name)
        return {
            "dataset": name,
            "n_spectra": 0,
            "error": "adapter yielded no spectra (missing data or import skip)",
            "elapsed_s": round(time.time() - t0, 1),
        }

    elements = sorted(elem_union)
    elem_idx = {el: i for i, el in enumerate(elements)}

    # Composition matrix: NaN = element not certified for that spectrum.
    comp_mat = np.full((n, len(elements)), np.nan, dtype=np.float64)
    for i, comp in enumerate(comp_maps):
        for el, wt in comp.items():
            comp_mat[i, elem_idx[el]] = float(wt)

    uniform = _grids_uniform(wls)
    grid_sizes = np.array([w.shape[0] for w in wls], dtype=np.int64)

    payload: dict = {
        "dataset": name,
        "spectrum_ids": np.array(spectrum_ids),
        "uniform_grid": np.bool_(uniform),
        "elements": np.array(elements),
        "composition_wt": comp_mat,
        "composition_basis": np.array(bases),
        "resolving_power": np.array(rps, dtype=np.float64),
        "notes": np.array(notes, dtype=object),
        "grid_sizes": grid_sizes,
    }
    if uniform:
        payload["wavelength_nm"] = wls[0]
        payload["intensity"] = np.vstack(intens)
    else:
        payload["wavelength_nm"] = np.array(wls, dtype=object)
        payload["intensity"] = np.array(intens, dtype=object)

    out_path = out_dir / f"{name}.npz"
    np.savez_compressed(out_path, **payload)

    elapsed = round(time.time() - t0, 1)
    man = {
        "dataset": name,
        "n_spectra": int(n),
        "elements": elements,
        "n_elements": len(elements),
        "uniform_grid": bool(uniform),
        "grid_size": int(grid_sizes[0]) if uniform else None,
        "grid_size_min": int(grid_sizes.min()),
        "grid_size_max": int(grid_sizes.max()),
        "composition_basis_counts": {
            b: int((np.array(bases) == b).sum()) for b in sorted(set(bases))
        },
        "wl_min_nm": float(min(w.min() for w in wls)),
        "wl_max_nm": float(max(w.max() for w in wls)),
        "n_with_composition": int(sum(1 for c in comp_maps if c)),
        "npz_path": str(out_path),
        "npz_bytes": out_path.stat().st_size,
        "tier": entry.tier,
        "elapsed_s": elapsed,
    }
    log.info(
        "%s: %d spectra, grid %s (%d-%d), %d elements, %s, %.1fs -> %s",
        name,
        n,
        "uniform" if uniform else "ragged",
        grid_sizes.min(),
        grid_sizes.max(),
        len(elements),
        man["composition_basis_counts"],
        elapsed,
        out_path,
    )
    return man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp/rtval/data")
    ap.add_argument("--datasets", nargs="*", default=TARGET_DATASETS)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_default_datasets()
    log.info("Registered datasets: %s", [e.name for e in iter_datasets()])

    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cflibs_file": __import__("cflibs").__file__,
        "datasets": {},
    }
    for name in args.datasets:
        log.info("=== exporting %s ===", name)
        try:
            man = export_dataset(name, out_dir)
        except Exception as exc:  # noqa: BLE001
            log.exception("%s: FAILED", name)
            man = {"dataset": name, "error": f"{type(exc).__name__}: {exc}"}
        manifest["datasets"][name] = man

    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote manifest -> %s", man_path)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
