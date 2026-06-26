"""Curated emission-line lists per alloy element (DED-PLAN step 4).

The benchmark bypasses peak detection: it extracts intensities at KNOWN line
positions. This module selects, per element, a set of strong, well-isolated,
energy-spread lines from the atomic DB so the constrained Boltzmann fit is
well-conditioned (each element needs >=2 lines spanning ~>1.5 eV in E_k).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .alloy_definitions import ALLOY_WINDOWS_NM, elements_of

K_B_EV = 8.617333262e-5  # Boltzmann constant, eV/K


@dataclass
class LineSpec:
    element: str
    ionization_stage: int
    wavelength_nm: float
    E_k_ev: float
    g_k: float
    A_ki: float
    aki_uncertainty: Optional[float] = None
    is_resonance: bool = False
    blended: bool = False


def _emissivity_weight(tr, T_K: float) -> float:
    """Relative upper-level population x A_ki x g_k (Boltzmann at T)."""
    return float(tr.g_k * tr.A_ki * np.exp(-tr.E_k_ev / (K_B_EV * T_K)))


def select_lines(
    db,
    element: str,
    window: Tuple[float, float],
    n_lines: int,
    T_K: float = 11000.0,
    stages: Sequence[int] = (1, 2),
    exclude_resonance: bool = True,
    min_separation_nm: float = 0.12,
    prefer_spread: bool = True,
) -> List[LineSpec]:
    """Pick up to ``n_lines`` strong, isolated, energy-spread lines for ``element``.

    Ranks candidates by Boltzmann-weighted emissivity at ``T_K``. With
    ``prefer_spread`` (default), it then bins the strong-line pool by upper-level
    energy E_k and takes the strongest isolated line per bin, so the chosen set
    spans a wide E_k range (a clustered E_k gives a poorly-constrained Boltzmann
    slope -> bad T). Resonance lines are excluded by default (they self-absorb);
    if that leaves too few, resonance lines are admitted as a fallback.
    """
    wmin, wmax = window

    def _gather(allow_resonance: bool) -> List:
        out = []
        for stage in stages:
            for tr in db.get_transitions(
                element, ionization_stage=stage, wavelength_min=wmin, wavelength_max=wmax
            ):
                if not tr.A_ki or tr.A_ki <= 0 or not tr.g_k or tr.g_k <= 0:
                    continue
                if tr.E_k_ev is None:
                    continue
                if not allow_resonance and getattr(tr, "is_resonance", False):
                    continue
                out.append(tr)
        out.sort(key=lambda t: _emissivity_weight(t, T_K), reverse=True)
        return out

    cands = _gather(allow_resonance=not exclude_resonance)
    if exclude_resonance and len(cands) < max(2, n_lines // 2):
        cands = _gather(allow_resonance=True)  # fallback: too few non-resonance

    def _isolated(tr, picked) -> bool:
        return all(abs(tr.wavelength_nm - c.wavelength_nm) >= min_separation_nm for c in picked)

    chosen: List = []
    pool = cands[: max(n_lines * 4, n_lines)]
    eks = np.array([t.E_k_ev for t in pool], dtype=float)
    if prefer_spread and len(pool) > n_lines and eks.size and float(eks.max() - eks.min()) > 0:
        # bin the strong pool by E_k; take the strongest isolated line per bin
        edges = np.linspace(eks.min(), eks.max(), n_lines + 1)
        used: set = set()
        for b in range(n_lines):
            lo, hi = edges[b], edges[b + 1]
            last = b == n_lines - 1
            for i, tr in enumerate(pool):
                if i in used:
                    continue
                ek = tr.E_k_ev
                if (lo <= ek < hi) or (last and ek <= hi):
                    if _isolated(tr, chosen):
                        chosen.append(tr)
                        used.add(i)
                        break
        # fill any remaining slots with the strongest unused isolated lines
        for i, tr in enumerate(pool):
            if len(chosen) >= n_lines:
                break
            if i not in used and _isolated(tr, chosen):
                chosen.append(tr)
                used.add(i)
    else:
        for tr in cands:
            if _isolated(tr, chosen):
                chosen.append(tr)
            if len(chosen) >= n_lines:
                break

    return [
        LineSpec(
            element=element,
            ionization_stage=int(tr.ionization_stage),
            wavelength_nm=float(tr.wavelength_nm),
            E_k_ev=float(tr.E_k_ev),
            g_k=float(tr.g_k),
            A_ki=float(tr.A_ki),
            aki_uncertainty=getattr(tr, "aki_uncertainty", None),
            is_resonance=bool(getattr(tr, "is_resonance", False)),
        )
        for tr in chosen
    ]


#: Per-element line budgets (DED-PLAN: Ti~12, Al~8, V~6; others ~10).
DEFAULT_LINE_BUDGET: Dict[str, int] = {"Ti": 12, "Al": 8, "V": 6}


def build_alloy_line_list(
    db,
    alloy: str,
    T_K: float = 11000.0,
    budget: Optional[Dict[str, int]] = None,
    blend_tol_nm: float = 0.05,
) -> Dict[str, List[LineSpec]]:
    """Build the per-element line list for an alloy and flag cross-element blends.

    A line within ``blend_tol_nm`` of any other element's line is marked
    ``blended=True`` (the extractor/solver can down-weight or skip it).
    """
    budget = budget or {}
    window = ALLOY_WINDOWS_NM[alloy]
    per_el: Dict[str, List[LineSpec]] = {}
    for el in elements_of(alloy):
        n = budget.get(el, DEFAULT_LINE_BUDGET.get(el, 10))
        per_el[el] = select_lines(db, el, window, n, T_K=T_K)

    # cross-element blend flagging
    allspecs = [s for specs in per_el.values() for s in specs]
    for s in allspecs:
        for o in allspecs:
            if o is s or o.element == s.element:
                continue
            if abs(o.wavelength_nm - s.wavelength_nm) < blend_tol_nm:
                s.blended = True
                break
    return per_el
