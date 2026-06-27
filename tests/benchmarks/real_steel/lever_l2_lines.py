"""Lever L2 — physically-correct line-selection policy (real-steel gate).

Root cause (docs/research/real-steel-accuracy-levers.md): the baseline line
selection lets trace minors be observed **only via singly-ionized lines** at a
low fitted plasma T. The Saha ion->total back-correction
``N_I ∝ N_II · N_e · exp(E_ion / kT)`` then explodes at low T and the minor
soaks the closure (Cu 0.2 wt% recovered as ~93%).

L2 fixes this with a neutral-anchored selection policy:

(a) Per element, prefer **neutral** (ionization_stage == 1) lines with a wide
    upper-energy (E_k) spread, so the Saha correction is applied on (or very
    near) the neutral plane instead of being extrapolated up from ion lines.
(b) If an element has too few usable non-resonance neutral lines in band, admit
    its strong neutral **resonance** lines as an anchor (e.g. Cu I 324.75 /
    327.40 nm) rather than falling back to ion lines. If there is **no** neutral
    line at all in band, the element is dropped from the closure (better to omit
    a minor than let an ion-only Saha extrapolation soak it).
(c) Resonance lines (self-absorbing) are excluded for any element that has >=2
    strong non-resonance neutral lines, so self-absorbed majors are avoided
    where a non-resonance alternative exists.

The solve mirrors ``baseline_solve`` (extract at known positions -> constrained
Saha-Boltzmann graph solver with injected n_e) but swaps the line list.
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
from tests.benchmarks.real_steel.harness import run_benchmark  # noqa: E402

K_B_EV = 8.617333262e-5  # Boltzmann constant, eV/K

# Selection temperature for emissivity ranking. Deliberately a fixed, physically
# reasonable steel-plasma value (NOT fit per sample, NOT tuned to any truth).
SELECT_T_K = 8000.0
# A neutral resonance line is admitted as an anchor only when the strongest
# neutral line is resonance and beats the best non-resonance neutral line by
# this factor (i.e. the element is effectively neutral-resonance-only in
# practice; weak non-resonance neutrals will not extract from real spectra).
RESONANCE_ANCHOR_RATIO = 8.0


def _emis(tr, T_K: float) -> float:
    return float(tr.g_k * tr.A_ki * np.exp(-tr.E_k_ev / (K_B_EV * T_K)))


def _gather(db, element: str, stage: int, window: Tuple[float, float]) -> List:
    out = []
    for tr in db.get_transitions(
        element, ionization_stage=stage, wavelength_min=window[0], wavelength_max=window[1]
    ):
        if not tr.A_ki or tr.A_ki <= 0 or not tr.g_k or tr.g_k <= 0:
            continue
        if tr.E_k_ev is None:
            continue
        out.append(tr)
    out.sort(key=lambda t: _emis(t, SELECT_T_K), reverse=True)
    return out


def _spread_pick(cands: List, n_lines: int, min_separation_nm: float) -> List:
    """Take up to ``n_lines`` strong, isolated lines spanning a wide E_k range."""

    def _isolated(tr, picked) -> bool:
        return all(abs(tr.wavelength_nm - c.wavelength_nm) >= min_separation_nm for c in picked)

    pool = cands[: max(n_lines * 4, n_lines)]
    eks = np.array([t.E_k_ev for t in pool], dtype=float)
    chosen: List = []
    if len(pool) > n_lines and eks.size and float(eks.max() - eks.min()) > 0:
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
    return chosen


def select_l2_lines(
    db,
    element: str,
    window: Tuple[float, float],
    n_lines: int = 8,
    min_separation_nm: float = 0.12,
) -> List[LineSpec]:
    """Neutral-anchored, wide-E_k line selection for one element.

    Returns an empty list when the element has no usable neutral line in band
    (so it is dropped from the closure rather than observed ion-only).
    """
    neutral = _gather(db, element, 1, window)
    nonres = [t for t in neutral if not getattr(t, "is_resonance", False)]
    res = [t for t in neutral if getattr(t, "is_resonance", False)]

    if len(nonres) >= 2:
        pool = list(nonres)
        # (b)/(c): if the strongest neutral line overall is a much stronger
        # resonance line, the element is effectively resonance-only in real
        # spectra -- admit the top resonance neutral lines as an anchor so the
        # element is observed on the neutral plane instead of via ion lines.
        if res and neutral and getattr(neutral[0], "is_resonance", False):
            best_res = _emis(res[0], SELECT_T_K)
            best_nonres = _emis(nonres[0], SELECT_T_K)
            if best_nonres <= 0 or best_res / best_nonres >= RESONANCE_ANCHOR_RATIO:
                pool = list(neutral)  # resonance + non-resonance neutral
    elif neutral:
        pool = list(neutral)  # too few non-resonance: admit resonance anchor
    else:
        return []  # no neutral line in band -> drop element from closure

    chosen = _spread_pick(pool, n_lines, min_separation_nm)
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


def solve_l2(db, wl, intensity, truth) -> Dict[str, float]:
    """L2 solve: neutral-anchored line selection + constrained Saha-Boltzmann.

    Mirrors ``baseline_solve`` but uses :func:`select_l2_lines` so trace minors
    are anchored on neutral lines (never observed ion-only).
    """
    els = list(truth.keys())
    window = (float(wl.min()), float(wl.max()))
    specs: List[LineSpec] = []
    for e in els:
        specs.extend(select_l2_lines(db, e, window, 8))
    if not specs:
        return {}
    obs = extract_line_intensities(wl, intensity, specs, instrument_fwhm_nm=0.2)
    res = run_constrained_solver(db, obs, 1e17)
    return recovered_wt(res)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    r = run_benchmark(solve_l2, limit=a.limit)
    print("L2 (neutral-anchor) real-steel RMSEP (wt%):")
    for k in sorted(r):
        v = r[k]
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
