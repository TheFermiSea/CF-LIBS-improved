"""Nominal compositions + composition series for the DED precision benchmark.

All compositions are MASS fractions in wt% over a CONSTRAINED, KNOWN element
set (no oxygen, no oxides) and sum to exactly 100. The constrained set is the
key DED simplification: we solve only for the alloy's known species.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

#: Nominal feedstock compositions (wt%), constrained known element set, sum=100.
ALLOY_COMPOSITIONS: Dict[str, Dict[str, float]] = {
    # Ti-6Al-4V (Grade 5): the primary DED target.
    "Ti-6Al-4V": {"Ti": 90.0, "Al": 6.0, "V": 4.0},
    # Inconel 625 constrained to its 4 majors, renormalized to 100.
    "Inconel625": {"Ni": 64.5, "Cr": 22.5, "Mo": 9.5, "Nb": 3.5},
    # 316L stainless constrained to its 4 majors (Fe as matrix), renormalized.
    "316L": {"Fe": 68.0, "Cr": 17.0, "Ni": 12.0, "Mo": 3.0},
}

#: Suggested wavelength window (nm) per alloy for the forward model.
ALLOY_WINDOWS_NM: Dict[str, tuple] = {
    "Ti-6Al-4V": (250.0, 500.0),
    "Inconel625": (280.0, 520.0),
    "316L": (250.0, 500.0),
}


def make_series(base: Dict[str, float], element: str, values: np.ndarray) -> List[Dict[str, float]]:
    """Vary ``element`` across ``values`` (wt%), scaling the other elements
    proportionally so each composition still sums to exactly 100 wt%.

    This mimics DED drift: one species evaporates/enriches while the remaining
    elements keep their mutual ratios.
    """
    if element not in base:
        raise KeyError(f"{element!r} not in base composition {sorted(base)}")
    others = {e: w for e, w in base.items() if e != element}
    others_sum = sum(others.values())
    series: List[Dict[str, float]] = []
    for v in values:
        v = float(v)
        rest = 100.0 - v
        if rest <= 0:
            raise ValueError(f"{element}={v} leaves no mass for the rest")
        scaled = {e: w / others_sum * rest for e, w in others.items()}
        comp = {element: v}
        comp.update(scaled)
        series.append(comp)
    return series


#: Per-alloy composition scans (the drift axes from DED-PLAN section 4).
COMPOSITION_SERIES: Dict[str, Dict[str, List[Dict[str, float]]]] = {
    "Ti-6Al-4V": {
        "Al": make_series(ALLOY_COMPOSITIONS["Ti-6Al-4V"], "Al", np.arange(4.0, 8.01, 0.5)),
        "V": make_series(ALLOY_COMPOSITIONS["Ti-6Al-4V"], "V", np.arange(3.0, 6.01, 0.5)),
    },
    "Inconel625": {
        "Cr": make_series(ALLOY_COMPOSITIONS["Inconel625"], "Cr", np.arange(18.0, 25.01, 1.0)),
        "Mo": make_series(ALLOY_COMPOSITIONS["Inconel625"], "Mo", np.arange(6.0, 11.01, 1.0)),
    },
    "316L": {
        "Cr": make_series(ALLOY_COMPOSITIONS["316L"], "Cr", np.arange(14.0, 19.01, 1.0)),
        "Ni": make_series(ALLOY_COMPOSITIONS["316L"], "Ni", np.arange(10.0, 14.01, 1.0)),
    },
}


def elements_of(alloy: str) -> List[str]:
    """The constrained known element set for an alloy."""
    return list(ALLOY_COMPOSITIONS[alloy].keys())
