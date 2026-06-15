#!/usr/bin/env python
"""Fe-separability analysis for candidate ID-benchmark spectral windows.

For each non-Fe candidate element, computes the fraction of its strongest lines
(by aki*gk) whose wavelength is NOT within half a resolution element of a
*comparably strong* Fe line at the chosen resolving power. A window that
maximizes the count of elements with diagnostic, Fe-separable lines is a window
where element identification is actually possible (not Fe-aliased).

Root cause this addresses: the ak3.1.3 corpus (224-265 nm deep-UV iron-group
forest) leaves 0-2/10 non-Fe panel elements with Fe-separable strong lines, so
an ideal detector caps at F1 ~ 0.31. See
docs/audit/2026-06-15-id-f1-rootcause.md.

Physics-only: numpy + sqlite3 only. Read-only against the atomic DB.

Usage::

    PYTHONPATH=$PWD python scripts/analyze_window_separability.py \
        --db ASD_da/libs_production.db --rp 1000
"""

from __future__ import annotations

import argparse
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_PANEL_NONFE = ["Al", "Co", "Cr", "Cu", "Mg", "Mn", "Ni", "Si", "Ti", "V"]

# Candidate windows, each a list of (lo, hi) nm segments.
DEFAULT_WINDOWS: Dict[str, List[Tuple[float, float]]] = {
    "224-265 (corpus baseline)": [(224.0, 265.0)],
    "285-850": [(285.0, 850.0)],
    "385-850 (recommended)": [(385.0, 850.0)],
    "240-340 + 385-850 (2seg)": [(240.0, 340.0), (385.0, 850.0)],
    "260-850 full": [(260.0, 850.0)],
}


def _lines_in_segments(
    cur: sqlite3.Cursor,
    element: str,
    segments: Sequence[Tuple[float, float]],
    sp_max: int = 2,
) -> List[Tuple[float, float]]:
    """Return [(wavelength_nm, strength=aki*gk)] for an element across segments."""
    out: List[Tuple[float, float]] = []
    for lo, hi in segments:
        cur.execute(
            "SELECT wavelength_nm, aki, gk FROM lines WHERE element=? AND sp_num<=? "
            "AND wavelength_nm BETWEEN ? AND ? AND aki>0 AND gk>0",
            (element, sp_max, float(lo), float(hi)),
        )
        out.extend((w, a * g) for (w, a, g) in cur.fetchall() if w)
    return out


def separable_fraction(
    cur: sqlite3.Cursor,
    element: str,
    segments: Sequence[Tuple[float, float]],
    resolving_power: float,
    top_n: int = 40,
    mask_ratio: float = 0.3,
) -> Optional[float]:
    """Fraction of an element's strongest lines that are Fe-separable.

    A candidate line at wavelength w is "Fe-separable" if no Fe line with
    strength >= mask_ratio * (candidate strength) lies within half a resolution
    element (0.5 * w / resolving_power) of it.
    """
    cand = _lines_in_segments(cur, element, segments)
    if not cand:
        return None
    cand.sort(key=lambda x: x[1], reverse=True)
    cand = cand[:top_n]
    fe = _lines_in_segments(cur, "Fe", segments)
    fe_w = np.array([w for w, _ in fe])
    fe_s = np.array([s for _, s in fe])
    n_sep = 0
    for w, cs in cand:
        tol = 0.5 * w / resolving_power
        if fe_w.size == 0:
            n_sep += 1
            continue
        near = np.abs(fe_w - w) <= tol
        if not (near & (fe_s >= mask_ratio * cs)).any():
            n_sep += 1
    return n_sep / len(cand)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="ASD_da/libs_production.db")
    ap.add_argument("--rp", type=float, default=1000.0, help="resolving power")
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument(
        "--mask-ratio",
        type=float,
        default=0.3,
        help="Fe line must be >= this * candidate strength to mask it",
    )
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    header = f"{'window':32s} | " + " ".join(f"{e:>4s}" for e in DEFAULT_PANEL_NONFE)
    header += " | n>=50%"
    print(f"RP={args.rp:g}, top_n={args.top_n}, mask_ratio={args.mask_ratio}")
    print(header)
    print("-" * len(header))
    for name, segs in DEFAULT_WINDOWS.items():
        cells: List[str] = []
        n_diag = 0
        for e in DEFAULT_PANEL_NONFE:
            f = separable_fraction(cur, e, segs, args.rp, args.top_n, args.mask_ratio)
            if f is None:
                cells.append(" na ")
            else:
                cells.append(f"{f * 100:3.0f}%")
                if f >= 0.5:
                    n_diag += 1
        print(f"{name:32s} | " + " ".join(cells) + f" | {n_diag:2d}/10")

    con.close()


if __name__ == "__main__":
    main()
