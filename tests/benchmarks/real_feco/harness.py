"""Real Fe-Co benchmark harness -- pluggable solve_fn, per-element RMSEP.

Cross-matrix generality gate for the shipped known-matrix/OPC mode: the same
``solve_fn(db, wl, intensity, truth) -> {element -> wt%}`` contract and
per-element + overall RMSEP scoring as :mod:`tests.benchmarks.real_steel.harness`,
but on the figshare 21984989 Fe-Co ladder (CEITEC, J. Vrabel; MIT) instead of the
steel coupon set. Used to test whether the real-steel OPC win generalizes to a
2-element near-binary alloy system spanning the full 0-100 wt% range.

Data layout (``data/real_feco/``, gitignored):

* ``labtrace_avantes_15mJ.h5`` -- ``f['spectra']`` (550, 4094) float32 = 11 samples
  x 50 shots; ``f['wavelengths']`` (4094,) over 241.6-411.5 nm (matches the steel
  pipeline window); ``f['spectra'].attrs['samples']`` = per-row labels '00'..'10'.
* ``sample_composition.xlsx`` -- certified wt% (real header on row 2: Sample, Fe,
  Co, Mn, Pb); a Fe<->Co 10 wt% ladder (99.6/0 -> 0.1/99.9) with trace Mn/Pb.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Fe-Co are the only quantified elements: trace Mn/Pb (<=0.2 wt%) are dropped and
# the truth is renormalized over Fe+Co (the closed basis the solver reports on).
FECO_ELEMENTS = ("Fe", "Co")
DATA_DIR = "data/real_feco"
H5_PATH = f"{DATA_DIR}/labtrace_avantes_15mJ.h5"
XLSX_PATH = f"{DATA_DIR}/sample_composition.xlsx"


def _load_truth_table(xlsx_path: str) -> Dict[int, Dict[str, float]]:
    """Parse the certified Fe-Co wt% per sample (0..10) from the xlsx.

    The real header is on row index 2 (``Sample, Fe, Co, Mn, Pb`` in columns 1-5);
    sample rows follow. Missing (NaN) Fe/Co cells are read as 0 wt%.
    """
    import pandas as pd

    raw = pd.read_excel(xlsx_path, sheet_name=0, header=None)
    # Locate the header row (the one whose cells contain 'Sample' and 'Fe').
    header_row = None
    for r in range(len(raw)):
        cells = [str(c).strip() for c in raw.iloc[r].tolist()]
        if "Sample" in cells and "Fe" in cells:
            header_row = r
            break
    if header_row is None:
        raise ValueError(f"could not find Sample/Fe header row in {xlsx_path}")
    cols = {str(c).strip(): i for i, c in enumerate(raw.iloc[header_row].tolist())}
    truth: Dict[int, Dict[str, float]] = {}
    for r in range(header_row + 1, len(raw)):
        sample_cell = raw.iloc[r, cols["Sample"]]
        if pd.isna(sample_cell):
            continue
        try:
            sid = int(float(sample_cell))
        except (TypeError, ValueError):
            continue
        row = {}
        for el in FECO_ELEMENTS:
            v = raw.iloc[r, cols[el]] if el in cols else np.nan
            row[el] = 0.0 if pd.isna(v) else float(v)
        truth[sid] = row
    return truth


def load_real_feco(h5_path: str = H5_PATH, xlsx_path: str = XLSX_PATH, min_wt: float = 0.05):
    """Yield (sample_id, wavelength_nm, mean_intensity, truth_wt) per Fe-Co sample.

    ``mean_intensity`` is the per-bin mean over the 50 shots of the sample
    (clipped at 0). ``truth_wt`` is the certified Fe/Co wt% present above
    ``min_wt``, renormalized to 100% over the modeled set (same closed basis the
    solver reports, identical convention to ``load_real_steel``).
    """
    import h5py

    truth_table = _load_truth_table(xlsx_path)
    with h5py.File(h5_path, "r") as f:
        spectra = np.asarray(f["spectra"][:], dtype=float)
        wl = np.asarray(f["wavelengths"][()], dtype=float)
        labels = np.asarray(f["spectra"].attrs["samples"]).astype(str)

    order = sorted(set(labels), key=lambda s: int(s))
    for label in order:
        sid = int(label)
        mask = labels == label
        inten = np.clip(spectra[mask].mean(axis=0), 0.0, None)
        cert = truth_table.get(sid, {})
        truth = {e: v for e, v in cert.items() if v is not None and float(v) > min_wt}
        tot = sum(truth.values())
        if tot <= 0:
            continue
        truth_n = {e: v / tot * 100.0 for e, v in truth.items()}
        yield label, wl, inten, truth_n


def score(results: List[Tuple[Dict[str, float], Dict[str, float]]]) -> Dict[str, float]:
    """Per-element + overall RMSEP (wt%) over (truth, pred) pairs (predictions renormalized).

    Identical convention to :func:`tests.benchmarks.real_steel.harness.score`.
    """
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
    out["rmsep_overall"] = (
        float(np.sqrt(np.mean(np.array(overall) ** 2))) if overall else float("nan")
    )
    out["n_samples"] = len(results)
    return out


def run_benchmark(
    solve_fn: Callable, db_path: str = "ASD_da/libs_production.db", limit: int | None = None
) -> Dict[str, float]:
    """Score ``solve_fn`` on the Fe-Co set. ``solve_fn(db, wl, intensity, truth)->{el:wt%}``."""
    import warnings

    warnings.filterwarnings("ignore")
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(db_path)
    results: List[Tuple[Dict[str, float], Dict[str, float]]] = []
    verbose = os.environ.get("REALFECO_QUIET", "") != "1"
    for i, (sid, wl, inten, truth) in enumerate(load_real_feco()):
        if limit is not None and i >= limit:
            break
        try:
            pred = solve_fn(db, wl, inten, truth)
        except Exception:  # noqa: BLE001 -- a failed solve scores as all-nan, not a crash
            pred = {}
        results.append((truth, pred or {}))
        if verbose:
            print(f"  [{i + 1}] sample {sid} done", flush=True)
    return score(results)
