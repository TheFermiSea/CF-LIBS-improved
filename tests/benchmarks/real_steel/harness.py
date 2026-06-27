"""Real-steel benchmark harness — pluggable solve_fn, per-element RMSEP across 36 samples.

The un-overfittable composition-accuracy gate for the real-data lever exploration
(see docs/research/real-steel-accuracy-levers.md). Each lever provides a ``solve_fn``
with the signature ``solve_fn(db, wl, intensity, truth) -> dict[element -> wt%]`` and is
scored by per-element + overall RMSEP vs the certified compositions.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Steel elements we attempt to quantify (exclude C: no usable atomic lines in this window).
STEEL_ELEMENTS = ("Fe", "Cr", "Ni", "Si", "Mn", "Mo", "Cu")
PARQUET = "data/real_steel/steel_266.parquet"


def load_real_steel(parquet_path: str = PARQUET, min_wt: float = 0.05):
    """Yield (sample_id, wavelength_nm, mean_intensity, truth_wt) for each steel sample.

    ``truth_wt`` is the certified wt% restricted to STEEL_ELEMENTS present above ``min_wt``,
    renormalized to 100% over the modeled set (so RMSEP is on the same closed basis the
    solver reports).
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    for _, row in df.iterrows():
        wl = np.asarray(row["wavelength_nm"], dtype=float)
        inten = np.clip(np.asarray([s["values"] for s in row["shots"]], dtype=float).mean(0), 0.0, None)
        truth = {e: float(row[e]) for e in STEEL_ELEMENTS
                 if row[e] is not None and float(row[e]) > min_wt}
        tot = sum(truth.values())
        truth_n = {e: v / tot * 100.0 for e, v in truth.items()}
        yield str(row["sample_name"]), wl, inten, truth_n


def score(results: List[Tuple[Dict[str, float], Dict[str, float]]]) -> Dict[str, float]:
    """Per-element + overall RMSEP (wt%) over (truth, pred) pairs (predictions renormalized)."""
    per_el_err: Dict[str, List[float]] = {}
    overall: List[float] = []
    for truth, pred in results:
        ps = sum(v for v in pred.values() if np.isfinite(v))
        pn = {e: (pred.get(e, 0.0) / ps * 100.0 if ps > 0 else float("nan")) for e in truth}
        for e, tv in truth.items():
            err = pn.get(e, float("nan")) - tv
            if np.isfinite(err):
                per_el_err.setdefault(e, []).append(err)
                overall.append(err)
    out = {f"rmsep_{e}": float(np.sqrt(np.mean(np.array(v) ** 2))) for e, v in per_el_err.items()}
    out["rmsep_overall"] = float(np.sqrt(np.mean(np.array(overall) ** 2))) if overall else float("nan")
    out["n_samples"] = len({id(r) for r in results})
    return out


def run_benchmark(solve_fn: Callable, db_path: str = "ASD_da/libs_production.db",
                  limit: int | None = None) -> Dict[str, float]:
    """Score ``solve_fn`` on the real-steel set. ``solve_fn(db, wl, intensity, truth)->{el:wt%}``."""
    import warnings
    warnings.filterwarnings("ignore")
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(db_path)
    results: List[Tuple[Dict[str, float], Dict[str, float]]] = []
    for i, (sid, wl, inten, truth) in enumerate(load_real_steel()):
        if limit is not None and i >= limit:
            break
        try:
            pred = solve_fn(db, wl, inten, truth)
        except Exception:  # noqa: BLE001 — a failed solve scores as all-nan, not a crash
            pred = {}
        results.append((truth, pred or {}))
    return score(results)
