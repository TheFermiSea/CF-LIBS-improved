"""Lever L3 — one-point-calibration F(lambda) (real-steel gate; Cavalcanti 2013).

Root cause (docs/research/real-steel-accuracy-levers.md): the *absolute* closure
``Sigma C_s = 1`` is flawed for a known matrix — a ~1% error on the major matrix
element (Fe) maps to a huge *relative* error on the trace minors, and per-element
systematic biases (line A_ki errors, partition-function error, residual
self-absorption, instrument response) survive the closure. For a **known matrix**
(steel / Ti-alloy / DED — the real goal) the standard fix is One-Point
Calibration (OPC, Cavalcanti et al. 2013): use ONE matrix-matched certified
standard to derive an empirical multiplicative correction ``F`` and apply it to
the Boltzmann-plot intensities of every other (unknown) sample, keeping CF-LIBS'
T / n_e adaptivity.

This module implements the honest, un-overfittable form:

  1. Pick ONE steel sample as the standard. Run the L2 neutral-anchor CF-LIBS
     solve on it (no correction) -> recovered wt% ``C_rec``.
  2. Derive a **per-element** multiplicative correction ``F_e = C_true_e / C_rec_e``
     from the standard's *certified* composition only. Because each element's
     Boltzmann-plot intercept (and hence its recovered number density) scales
     linearly with a constant rescale of that element's line intensities, scaling
     element ``e``'s intensities by ``F_e`` and re-solving reproduces the
     standard's known composition after closure (in-sample RMSEP ~ 0 by
     construction). ``F`` is the classic OPC relative-sensitivity factor set.
  3. Apply that SAME ``F`` UNCHANGED to all OTHER 35 samples (scaling each
     observation's intensity by ``F`` for its element before the solve) and report
     the **held-out** ``rmsep_overall``. ``F`` is never re-fit per test sample.

A per-LINE variant (``per_line=True``) keys the correction by ``(element,
wavelength)`` instead of by element; since the line selection is spectrum-position
independent (depends only on the DB + the shared instrument window), the same
lines appear in every sample, so per-line F is portable. Per-element is the
default — it is the most robust (one factor per element, not per noisy line) and
is what reproduces the certified composition.

HONEST EVALUATION: ``F`` is derived from the standard's certified truth ONLY and
applied unchanged to the held-out set. We never tune ``F`` to any held-out
sample's truth. The headline is the held-out ``rmsep_overall``; we also sweep a
few different standards and report the spread.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from tests.benchmarks.ded_precision.benchmark_runner import (  # noqa: E402
    extract_line_intensities,
)
from tests.benchmarks.ded_precision.line_lists import LineSpec  # noqa: E402
from tests.benchmarks.ded_precision.solver_runner import (  # noqa: E402
    recovered_wt,
    run_constrained_solver,
)
from tests.benchmarks.real_steel.harness import (  # noqa: E402
    load_real_steel,
    score,
)
from tests.benchmarks.real_steel.lever_l2_lines import select_l2_lines  # noqa: E402

# Floors guarding the F = C_true / C_rec ratio when the baseline recovers ~0 for
# an element (degenerate ratio). A recovered/true wt% below this is treated as the
# floor so F stays finite; F itself is also clamped to a sane band.
_MIN_WT = 1e-3
_F_MIN, _F_MAX = 1e-2, 1e2


def _line_key(element: str, wavelength_nm: float) -> Tuple[str, float]:
    return (element, round(float(wavelength_nm), 3))


def _solve_with_F(
    db,
    wl,
    intensity,
    truth,
    F: Dict,
    *,
    per_line: bool = False,
) -> Dict[str, float]:
    """L2 solve with an OPC correction applied to the Boltzmann-plot intensities.

    Each observation's intensity is multiplied by its element's (or line's) factor
    ``F`` before the constrained Saha-Boltzmann solve. ``F`` empty -> plain L2.
    """
    els = list(truth.keys())
    window = (float(wl.min()), float(wl.max()))
    specs: List[LineSpec] = []
    for e in els:
        specs.extend(select_l2_lines(db, e, window, 8))
    if not specs:
        return {}
    obs = extract_line_intensities(wl, intensity, specs, instrument_fwhm_nm=0.2)
    for o in obs:
        if per_line:
            f = F.get(_line_key(o.element, o.wavelength_nm), 1.0)
        else:
            f = F.get(o.element, 1.0)
        o.intensity = float(o.intensity) * float(f)
        o.intensity_uncertainty = max(float(o.intensity_uncertainty) * float(f), 1e-12)
    res = run_constrained_solver(db, obs, 1e17)
    return recovered_wt(res)


def derive_F(db, standard, *, per_line: bool = False) -> Dict:
    """Derive the OPC correction from ONE certified standard.

    Per-element: ``F_e = C_true_e / C_rec_e`` (C_rec = uncorrected L2 recovery,
    renormalized to 100% over the standard's modeled element set, matching the
    closed basis the truth uses). Per-line: every line of element ``e`` inherits
    ``F_e`` (a constant per-element rescale is the OPC-faithful, closure-portable
    correction; a genuinely per-line factor would need a per-line reference
    intensity, which a single composition standard does not provide).
    """
    _sid, wl, inten, truth = standard
    rec = _solve_with_F(db, wl, inten, truth, {}, per_line=False)
    rec = {e: rec.get(e, 0.0) for e in truth}
    tot = sum(v for v in rec.values() if np.isfinite(v) and v > 0)
    F_el: Dict[str, float] = {}
    for e, tv in truth.items():
        cn = (rec[e] / tot * 100.0) if (tot > 0 and np.isfinite(rec[e])) else 0.0
        f = max(tv, _MIN_WT) / max(cn, _MIN_WT)
        F_el[e] = float(np.clip(f, _F_MIN, _F_MAX))
    if not per_line:
        return F_el
    # Per-line: replay the standard's line selection and tag each line with its
    # element factor (portable because selection is sample-independent).
    window = (float(wl.min()), float(wl.max()))
    F_line: Dict[Tuple[str, float], float] = {}
    for e in truth:
        for ls in select_l2_lines(db, e, window, 8):
            F_line[_line_key(e, ls.wavelength_nm)] = F_el.get(e, 1.0)
    return F_line


def run_opc(
    db_path: str = "ASD_da/libs_production.db",
    std_index: int = 0,
    *,
    per_line: bool = False,
    limit: int | None = None,
) -> Dict[str, object]:
    """Honest OPC: derive F on one standard, apply unchanged held-out, score."""
    import warnings

    warnings.filterwarnings("ignore")
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(db_path)
    samples = list(load_real_steel())
    if limit is not None:
        samples = samples[:limit]
    if std_index >= len(samples):
        std_index = 0
    standard = samples[std_index]

    print(f"[L3] standard = sample {standard[0]} (index {std_index}); deriving F ...", flush=True)
    F = derive_F(db, standard, per_line=per_line)

    # In-sample (context only): apply F back to the standard.
    in_pred = _solve_with_F(db, standard[1], standard[2], standard[3], F, per_line=per_line)
    in_rmsep = score([(standard[3], in_pred or {})])["rmsep_overall"]

    # Held-out headline: apply F UNCHANGED to every OTHER sample.
    held: List[Tuple[Dict[str, float], Dict[str, float]]] = []
    for i, (sid, wl, inten, truth) in enumerate(samples):
        if i == std_index:
            continue
        pred = _solve_with_F(db, wl, inten, truth, F, per_line=per_line)
        held.append((truth, pred or {}))
        print(f"  [held-out {len(held)}] sample {sid} done", flush=True)
    held_scores = score(held)

    return {
        "standard_sample": standard[0],
        "std_index": std_index,
        "per_line": per_line,
        "F": {str(k): v for k, v in F.items()},
        "standard_in_sample_rmsep": in_rmsep,
        "held_out": held_scores,
    }


def standard_spread(
    db_path: str = "ASD_da/libs_production.db",
    std_indices: Tuple[int, ...] = (0, 5, 10, 15, 20),
    *,
    per_line: bool = False,
    limit: int | None = None,
) -> List[Dict[str, object]]:
    """Run OPC for several different standards; report the held-out RMSEP spread."""
    runs: List[Dict[str, object]] = []
    for si in std_indices:
        out = run_opc(db_path, std_index=si, per_line=per_line, limit=limit)
        ho = out["held_out"]["rmsep_overall"]
        print(
            f"[L3 spread] standard={out['standard_sample']} (idx {si}) "
            f"held-out rmsep_overall={ho:.3f}",
            flush=True,
        )
        runs.append(out)
    return runs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="ASD_da/libs_production.db")
    ap.add_argument("--std-index", type=int, default=0)
    ap.add_argument("--per-line", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--spread",
        action="store_true",
        help="sweep several standards and report the held-out RMSEP spread",
    )
    ap.add_argument("--std-indices", type=str, default="0,5,10,15,20")
    a = ap.parse_args()

    if a.spread:
        idxs = tuple(int(x) for x in a.std_indices.split(",") if x.strip())
        runs = standard_spread(a.db_path, std_indices=idxs, per_line=a.per_line, limit=a.limit)
        hos = [float(r["held_out"]["rmsep_overall"]) for r in runs]
        print("\nL3 (OPC) held-out rmsep_overall across standards:")
        for r, ho in zip(runs, hos):
            print(f"  standard {r['standard_sample']} (idx {r['std_index']}): {ho:.3f}")
        arr = np.array(hos, dtype=float)
        print(
            f"  spread: min={arr.min():.3f} median={np.median(arr):.3f} "
            f"max={arr.max():.3f} mean={arr.mean():.3f}"
        )
    else:
        out = run_opc(a.db_path, std_index=a.std_index, per_line=a.per_line, limit=a.limit)
        print("\nL3 (one-point-calibration F) real-steel results:")
        print(f"  standard: {out['standard_sample']} (index {out['std_index']})")
        print(f"  per_line: {out['per_line']}")
        print(f"  standard_in_sample_rmsep: {out['standard_in_sample_rmsep']:.3f}")
        print("  HELD-OUT RMSEP (wt%):")
        hs = out["held_out"]
        for k in sorted(hs):
            v = hs[k]
            print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
