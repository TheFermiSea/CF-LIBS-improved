"""Atomic-data sanity check for the Fe-Co cross-matrix benchmark.

Before claiming anything about OPC generality on Fe-Co, we must confirm the DB
actually *has* Fe and Co lines in the labtrace window (241-411 nm). OPC cannot
help recover an element the line list cannot see -- if Co is sparse/absent this is
the headline (a data-completeness limit, not an OPC limit).

Run::

    PYTHONPATH=$PWD python tests/benchmarks/real_feco/check_db_lines.py

Reports, per element x stage (neutral=1, singly-ionized=2), the number of
transitions in band and how many are *usable* (finite, positive A_ki and finite
E_k -- the fields the Boltzmann plot needs).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

WINDOW_NM: Tuple[float, float] = (241.6, 411.5)
ELEMENTS = ("Fe", "Co")
STAGES = (1, 2)
USABLE_MIN = 4  # min usable lines/element below which OPC has nothing to anchor on


def _is_usable(tr) -> bool:
    aki = getattr(tr, "A_ki", None)
    ek = getattr(tr, "E_k_ev", None)
    return (
        aki is not None
        and math.isfinite(float(aki))
        and float(aki) > 0.0
        and ek is not None
        and math.isfinite(float(ek))
    )


def count_lines(db_path: str = "ASD_da/libs_production.db") -> Dict[str, Dict]:
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(db_path)
    out: Dict[str, Dict] = {}
    for el in ELEMENTS:
        per_el = {"total_usable": 0, "stages": {}}
        for stage in STAGES:
            trs = db.get_transitions(
                el,
                ionization_stage=stage,
                wavelength_min=WINDOW_NM[0],
                wavelength_max=WINDOW_NM[1],
            )
            usable = [t for t in trs if _is_usable(t)]
            per_el["stages"][stage] = {"total": len(trs), "usable": len(usable)}
            per_el["total_usable"] += len(usable)
        out[el] = per_el
    return out


def main() -> None:
    res = count_lines()
    print(f"Fe-Co DB line check, window {WINDOW_NM[0]}-{WINDOW_NM[1]} nm")
    print("=" * 60)
    for el, info in res.items():
        print(f"{el}:  total usable (stages 1+2) = {info['total_usable']}")
        for stage, c in info["stages"].items():
            label = {1: "I (neutral)", 2: "II (singly-ionized)"}[stage]
            print(f"    stage {stage} {label:<22} total={c['total']:>4}  usable={c['usable']:>4}")
        verdict = "OK" if info["total_usable"] >= USABLE_MIN else "TOO FEW (OPC cannot anchor)"
        print(f"    -> {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
